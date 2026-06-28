package cowmap

import (
	"hash/maphash"
	"iter"
	"math/bits"
	"runtime"
	"sync/atomic"
)

var hashSeed = maphash.MakeSeed()

// numShards balances memory footprint and concurrent throughput.
// 64 shards strip CPU cache-line bouncing cleanly across high-core systems.
const numShards = 64
const shardMask = numShards - 1
const maxShift = 30

type entry[K comparable, V any] struct {
	key   K
	value V
}

type collisionNode[K comparable, V any] struct {
	entries []entry[K, V]
}

type node[K comparable, V any] struct {
	bitmap   uint32
	children []any
}

// Shard encapsulates an independent HAMT trie root to distribute CAS contention.
type shard[K comparable, V any] struct {
	root atomic.Pointer[node[K, V]]
}

// Map implements a highly concurrent, low-contention Sharded CoW HAMT.
type Map[K comparable, V any] struct {
	shards [numShards]shard[K, V]
	hasher func(K) uint32
}

// New instantiates an empty Sharded HAMT Map.
func New[K comparable, V any](customHasher func(K) uint32) *Map[K, V] {
	m := &Map[K, V]{}
	for i := 0; i < numShards; i++ {
		m.shards[i].root.Store(&node[K, V]{bitmap: 0, children: nil})
	}
	if customHasher != nil {
		m.hasher = customHasher
	} else {
		m.hasher = func(key K) uint32 {
			return uint32(maphash.Comparable(hashSeed, key))
		}
	}
	return m
}

// --- Snapshot Reader Operations (Zero Contention across Shards) ---

// Get performs zero-allocation data resolution on a constant shard snapshot.
func (m *Map[K, V]) Get(key K) (V, bool) {
	hash := m.hasher(key)
	shardIdx := hash & shardMask

	curr := m.shards[shardIdx].root.Load()
	shift := uint(5) // start at 5 since first bits determined the shard

	for curr != nil {
		idx := (hash >> shift) & 0x1F
		mask := uint32(1) << idx

		if (curr.bitmap & mask) == 0 {
			break
		}

		pos := bits.OnesCount32(curr.bitmap & (mask - 1))
		child := curr.children[pos]

		if nextNode, ok := child.(*node[K, V]); ok {
			curr = nextNode
			shift += 5
			continue
		}

		if kv, ok := child.(entry[K, V]); ok {
			if kv.key == key {
				return kv.value, true
			}
			break
		}

		if col, ok := child.(*collisionNode[K, V]); ok {
			for _, e := range col.entries {
				if e.key == key {
					return e.value, true
				}
			}
			break
		}
	}

	var zero V
	return zero, false
}

// All exposes a native iterator over all elements across all shards sequentially.
func (m *Map[K, V]) All() iter.Seq2[K, V] {
	return func(yield func(K, V) bool) {
		for i := 0; i < numShards; i++ {
			snapshot := m.shards[i].root.Load()
			if !snapshot.iterate(yield) {
				return
			}
		}
	}
}

func (n *node[K, V]) iterate(yield func(K, V) bool) bool {
	if n == nil {
		return true
	}
	for _, child := range n.children {
		if nextNode, ok := child.(*node[K, V]); ok {
			if !nextNode.iterate(yield) {
				return false
			}
		} else if kv, ok := child.(entry[K, V]); ok {
			if !yield(kv.key, kv.value) {
				return false
			}
		} else if col, ok := child.(*collisionNode[K, V]); ok {
			for _, e := range col.entries {
				if !yield(e.key, e.value) {
					return false
				}
			}
		}
	}
	return true
}

// --- High-Performance Transactional Writer Operations ---

// Insert targets a specific internal shard root, reducing global CPU cache invalidation.
func (m *Map[K, V]) Insert(key K, value V) {
	hash := m.hasher(key)
	shardIdx := hash & shardMask
	targetShard := &m.shards[shardIdx]

	backoff := 1
	for {
		currentRoot := targetShard.root.Load()
		newRoot := currentRoot.insert(hash, 5, key, value, m.hasher)

		if targetShard.root.CompareAndSwap(currentRoot, newRoot) {
			return
		}
		executeBackoff(&backoff)
	}
}

// Delete drops an element out of its isolated sub-shard structure.
func (m *Map[K, V]) Delete(key K) bool {
	hash := m.hasher(key)
	shardIdx := hash & shardMask
	targetShard := &m.shards[shardIdx]

	backoff := 1
	for {
		currentRoot := targetShard.root.Load()
		newRoot, found := currentRoot.delete(hash, 5, key)
		if !found {
			return false
		}
		if newRoot == nil {
			newRoot = &node[K, V]{bitmap: 0, children: nil}
		}

		if targetShard.root.CompareAndSwap(currentRoot, newRoot) {
			return true
		}
		executeBackoff(&backoff)
	}
}

// --- HAMT Core Structural Mutations ---

func (n *node[K, V]) insert(hash uint32, shift uint, key K, value V, hasher func(K) uint32) *node[K, V] {
	idx := (hash >> shift) & 0x1F
	mask := uint32(1) << idx
	pos := bits.OnesCount32(n.bitmap & (mask - 1))

	cloned := n.clone()

	if (cloned.bitmap & mask) == 0 {
		cloned.bitmap |= mask
		cloned.children = append(cloned.children, nil)
		copy(cloned.children[pos+1:], cloned.children[pos:])
		cloned.children[pos] = entry[K, V]{key: key, value: value}
		return cloned
	}

	child := cloned.children[pos]

	if subNode, ok := child.(*node[K, V]); ok {
		cloned.children[pos] = subNode.insert(hash, shift+5, key, value, hasher)
		return cloned
	}

	if existingKV, ok := child.(entry[K, V]); ok {
		if existingKV.key == key {
			cloned.children[pos] = entry[K, V]{key: key, value: value}
			return cloned
		}

		if shift >= maxShift {
			cloned.children[pos] = &collisionNode[K, V]{
				entries: []entry[K, V]{existingKV, {key: key, value: value}},
			}
			return cloned
		}

		subNode := &node[K, V]{bitmap: 0, children: nil}
		subHash := hasher(existingKV.key)
		subNode = subNode.insert(subHash, shift+5, existingKV.key, existingKV.value, hasher)
		subNode = subNode.insert(hash, shift+5, key, value, hasher)

		cloned.children[pos] = subNode
		return cloned
	}

	if colNode, ok := child.(*collisionNode[K, V]); ok {
		newCol := &collisionNode[K, V]{
			entries: make([]entry[K, V], len(colNode.entries)),
		}
		copy(newCol.entries, colNode.entries)

		updated := false
		for i, e := range newCol.entries {
			if e.key == key {
				newCol.entries[i].value = value
				updated = true
				break
			}
		}
		if !updated {
			newCol.entries = append(newCol.entries, entry[K, V]{key: key, value: value})
		}
		cloned.children[pos] = newCol
		return cloned
	}

	return cloned
}

func (n *node[K, V]) delete(hash uint32, shift uint, key K) (*node[K, V], bool) {
	idx := (hash >> shift) & 0x1F
	mask := uint32(1) << idx

	if (n.bitmap & mask) == 0 {
		return n, false
	}

	pos := bits.OnesCount32(n.bitmap & (mask - 1))
	child := n.children[pos]

	cloned := n.clone()

	if subNode, ok := child.(*node[K, V]); ok {
		newSub, found := subNode.delete(hash, shift+5, key)
		if !found {
			return n, false
		}

		if newSub == nil || len(newSub.children) == 0 {
			cloned.bitmap &^= mask
			cloned.children = append(cloned.children[:pos], cloned.children[pos+1:]...)
		} else if len(newSub.children) == 1 {
			if singleKV, ok := newSub.children[0].(entry[K, V]); ok {
				cloned.children[pos] = singleKV
			} else {
				cloned.children[pos] = newSub
			}
		} else {
			cloned.children[pos] = newSub
		}

		if len(cloned.children) == 0 {
			return nil, true
		}
		return cloned, true
	}

	if kv, ok := child.(entry[K, V]); ok {
		if kv.key == key {
			cloned.bitmap &^= mask
			cloned.children = append(cloned.children[:pos], cloned.children[pos+1:]...)
			if len(cloned.children) == 0 {
				return nil, true
			}
			return cloned, true
		}
	}

	if colNode, ok := child.(*collisionNode[K, V]); ok {
		idxToRemove := -1
		for i, e := range colNode.entries {
			if e.key == key {
				idxToRemove = i
				break
			}
		}
		if idxToRemove == -1 {
			return n, false
		}

		if len(colNode.entries) == 2 {
			remIdx := 1 - idxToRemove
			cloned.children[pos] = colNode.entries[remIdx]
		} else {
			newCol := &collisionNode[K, V]{
				entries: make([]entry[K, V], 0, len(colNode.entries)-1),
			}
			newCol.entries = append(newCol.entries, colNode.entries[:idxToRemove]...)
			newCol.entries = append(newCol.entries, colNode.entries[idxToRemove+1:]...)
			cloned.children[pos] = newCol
		}
		return cloned, true
	}

	return n, false
}

// --- Allocation Optimizations ---

func (n *node[K, V]) clone() *node[K, V] {
	if n == nil {
		return nil
	}

	// Fast optimization: Pre-allocate capacity cleanly to minimize slice growing cost
	newChildren := make([]any, len(n.children), len(n.children)+1)
	copy(newChildren, n.children)
	return &node[K, V]{
		bitmap:   n.bitmap,
		children: newChildren,
	}
}

func executeBackoff(backoff *int) {
	// Active spin loop using runtime hints helps threads clear the CAS choke quickly
	for i := 0; i < *backoff; i++ {
		runtime.Gosched()
	}
	if *backoff < 32 {
		*backoff <<= 1
	}
}

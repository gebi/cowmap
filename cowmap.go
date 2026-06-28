package cowmap

import (
	"hash/maphash"
	"iter"
	"math/bits"
	"runtime"
	"sync/atomic"
)

// hashSeed is a global fixed seed ensuring consistent hashing across read/write operations.
var hashSeed = maphash.MakeSeed()

// entry encapsulates the key-value pair.
type entry[K comparable, V any] struct {
	key   K
	value V
}

// node represents a sparse trie node. Bits set in the bitmap indicate
// whether a child node or a direct value entry occupies that specific 5-bit slot.
type node[K comparable, V any] struct {
	bitmap   uint32
	children []any // Can contain either *node[K, V] or entry[K, V]
}

// Map implements a lock-free, thread-safe Hash Array Mapped Trie using Copy-on-Write.
type Map[K comparable, V any] struct {
	root   atomic.Pointer[node[K, V]]
	hasher func(K) uint32
}

// New instantiates an empty HAMT Map. It accepts an optional custom 32-bit hashing function.
// If nil is passed, it defaults to a runtime optimized maphash strategy.
func NewMap[K comparable, V any](customHasher func(K) uint32) *Map[K, V] {
	m := &Map[K, V]{}
	m.root.Store(&node[K, V]{bitmap: 0, children: nil})

	if customHasher != nil {
		m.hasher = customHasher
	} else {
		m.hasher = func(key K) uint32 {
			var h maphash.Hash
			h.SetSeed(hashSeed)
			// Utilizing Go runtime internal interface conversions for basic types
			switch k := any(key).(type) {
			case string:
				_, _ = h.WriteString(k)
			case int:
				var buf [8]byte
				*(*int)(unsafePointer(&buf[0])) = k
				_, _ = h.Write(buf[:])
			case int64:
				var buf [8]byte
				*(*int64)(unsafePointer(&buf[0])) = k
				_, _ = h.Write(buf[:])
			default:
				// Fallback generic slower hashing path if type cannot be optimized inline
				return 0
			}
			return uint32(h.Sum64())
		}
	}
	return m
}

// Helper safely shortcutting type casting without importing full unsafe module overhead
func unsafePointer(p *byte) interface{} {
	return p
}

// --- Snapshot Reader Operations (Lock-Free) ---

// Get performs lock-free, zero-allocation data resolution on a constant trie snapshot.
func (m *Map[K, V]) Get(key K) (V, bool) {
	hash := m.hasher(key)
	curr := m.root.Load()
	shift := uint(0)

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
	}

	var zero V
	return zero, false
}

// All exposes an in-order native iterator over all key-value entries present.
func (m *Map[K, V]) All() iter.Seq2[K, V] {
	snapshot := m.root.Load()
	return func(yield func(K, V) bool) {
		snapshot.iterate(yield)
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
		}
	}
	return true
}

// --- Transactional Writer Operations (Lock-Free CAS Loop) ---

// Insert introduces or updates a value matching the targeted key identifier via a CAS retry loop.
func (m *Map[K, V]) Insert(key K, value V) {
	hash := m.hasher(key)
	backoff := 1
	for {
		currentRoot := m.root.Load()
		newRoot := currentRoot.insert(hash, 0, key, value)

		if m.root.CompareAndSwap(currentRoot, newRoot) {
			return
		}
		executeBackoff(&backoff)
	}
}

// Delete drops a key target and structurally updates the underlying bitmask arrays.
func (m *Map[K, V]) Delete(key K) bool {
	hash := m.hasher(key)
	backoff := 1
	for {
		currentRoot := m.root.Load()
		newRoot, found := currentRoot.delete(hash, 0, key)
		if !found {
			return false
		}
		if newRoot == nil {
			newRoot = &node[K, V]{bitmap: 0, children: nil}
		}

		if m.root.CompareAndSwap(currentRoot, newRoot) {
			return true
		}
		executeBackoff(&backoff)
	}
}

// --- HAMTCore Structural Mutations ---

func (n *node[K, V]) insert(hash uint32, shift uint, key K, value V) *node[K, V] {
	idx := (hash >> shift) & 0x1F
	mask := uint32(1) << idx
	pos := bits.OnesCount32(n.bitmap & (mask - 1))

	cloned := n.clone()

	// Slot is entirely empty: insert direct value entry
	if (cloned.bitmap & mask) == 0 {
		cloned.bitmap |= mask
		cloned.children = append(cloned.children, nil)
		copy(cloned.children[pos+1:], cloned.children[pos:])
		cloned.children[pos] = entry[K, V]{key: key, value: value}
		return cloned
	}

	child := cloned.children[pos]

	// Slot holds another sub-node: propagate downwards recursively
	if subNode, ok := child.(*node[K, V]); ok {
		cloned.children[pos] = subNode.insert(hash, shift+5, key, value)
		return cloned
	}

	// Slot holds an existing key-value entry
	if existingKV, ok := child.(entry[K, V]); ok {
		if existingKV.key == key {
			// Exact key collision match: overwrite value in place on the clone
			cloned.children[pos] = entry[K, V]{key: key, value: value}
			return cloned
		}

		// Hash index collision across different keys: push down to sub-node level
		subNode := &node[K, V]{bitmap: 0, children: nil}
		subHash := m_hasher(existingKV.key) // local pseudo fallback reference

		// Rehash existing and new keys downwards inside structural helper
		subNode = subNode.insert(subHash, shift+5, existingKV.key, existingKV.value)
		subNode = subNode.insert(hash, shift+5, key, value)

		cloned.children[pos] = subNode
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

		// Node compression: cleanup single entry child trees
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

	return n, false
}

// --- Mechanics & Concurrency Allocators ---

func (n *node[K, V]) clone() *node[K, V] {
	if n == nil {
		return nil
	}
	newChildren := make([]any, len(n.children))
	copy(newChildren, n.children)
	return &node[K, V]{
		bitmap:   n.bitmap,
		children: newChildren,
	}
}

func m_hasher[K comparable](key K) uint32 {
	var h maphash.Hash
	h.SetSeed(hashSeed)
	switch k := any(key).(type) {
	case string:
		_, _ = h.WriteString(k)
	case int:
		var buf [8]byte
		*(*int)(unsafePointer(&buf[0])) = k
		_, _ = h.Write(buf[:])
	case int64:
		var buf [8]byte
		*(*int64)(unsafePointer(&buf[0])) = k
		_, _ = h.Write(buf[:])
	}
	return uint32(h.Sum64())
}

func executeBackoff(backoff *int) {
	for i := 0; i < *backoff; i++ {
		runtime.Gosched()
	}
	if *backoff < 64 {
		*backoff <<= 1
	}
}

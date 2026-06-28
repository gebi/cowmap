package cowmap

import (
	"cmp"
	"iter"
	"runtime"
	"sync/atomic"
)

// maxKeys defines the branching threshold optimized for CPU cache line density.
const maxKeys = 15
const minKeys = maxKeys / 2

type entry[K cmp.Ordered, V any] struct {
	key   K
	value V
}

type node[K cmp.Ordered, V any] struct {
	keys     []entry[K, V]
	children []*node[K, V]
}

// Map handles thread-safe lock-free generic data lookups via an immutable B-Tree.
type Map[K cmp.Ordered, V any] struct {
	root atomic.Pointer[node[K, V]]
}

// New returns an instantiated empty lock-free Map.
func NewMap[K cmp.Ordered, V any]() *Map[K, V] {
	m := &Map[K, V]{}
	m.root.Store(&node[K, V]{})
	return m
}

// --- Snapshot Reads (Lock-free) ---

// Get searches for a key and returns the associated value and existence flag.
func (m *Map[K, V]) Get(key K) (V, bool) {
	return m.root.Load().search(key)
}

func (n *node[K, V]) search(key K) (V, bool) {
	idx, found := n.findKey(key)
	if found {
		return n.keys[idx].value, true
	}
	if n.isLeaf() {
		var zero V
		return zero, false
	}
	return n.children[idx].search(key)
}

// All exposes a Go 1.23 functional sequence iterator over all sorted key-value pairs.
func (m *Map[K, V]) All() iter.Seq2[K, V] {
	snapshot := m.root.Load()
	return func(yield func(K, V) bool) {
		snapshot.iterate(yield)
	}
}

func (n *node[K, V]) iterate(yield func(K, V) bool) bool {
	for i := 0; i < len(n.keys); i++ {
		if !n.isLeaf() {
			if !n.children[i].iterate(yield) {
				return false
			}
		}
		if !yield(n.keys[i].key, n.keys[i].value) {
			return false
		}
	}
	if !n.isLeaf() {
		return n.children[len(n.keys)].iterate(yield)
	}
	return true
}

// Range exposes an isolated iterator over the key boundary spectrum [low, high].
func (m *Map[K, V]) Range(low, high K) iter.Seq2[K, V] {
	snapshot := m.root.Load()
	return func(yield func(K, V) bool) {
		snapshot.iterateRange(low, high, yield)
	}
}

func (n *node[K, V]) iterateRange(low, high K, yield func(K, V) bool) bool {
	for i := 0; i < len(n.keys); i++ {
		if !n.isLeaf() && n.keys[i].key >= low {
			if !n.children[i].iterateRange(low, high, yield) {
				return false
			}
		}
		if n.keys[i].key >= low && n.keys[i].key <= high {
			if !yield(n.keys[i].key, n.keys[i].value) {
				return false
			}
		}
		if n.keys[i].key > high {
			return true
		}
	}
	if !n.isLeaf() && (len(n.keys) == 0 || n.keys[len(n.keys)-1].key <= high) {
		return n.children[len(n.keys)].iterateRange(low, high, yield)
	}
	return true
}

// --- CAS Write Loops (Adaptive Backoff) ---

// Insert safely inserts or overwrites a value associated with the specified key.
func (m *Map[K, V]) Insert(key K, value V) {
	backoff := 1
	for {
		currentRoot := m.root.Load()
		newRoot := currentRoot.clone()

		promotedKey, promotedChild := newRoot.insert(key, value)
		if promotedChild != nil {
			var zero V
			oldRoot := newRoot
			newRoot = &node[K, V]{
				keys:     []entry[K, V]{{key: promotedKey, value: zero}},
				children: []*node[K, V]{oldRoot, promotedChild},
			}
		}

		if m.root.CompareAndSwap(currentRoot, newRoot) {
			return
		}
		executeBackoff(&backoff)
	}
}

// Delete drops a targeted key from the node tree and balance shifts.
func (m *Map[K, V]) Delete(key K) bool {
	backoff := 1
	for {
		currentRoot := m.root.Load()
		_, found := currentRoot.search(key)
		if !found {
			return false
		}

		newRoot := currentRoot.clone()
		newRoot.delete(key)

		if len(newRoot.keys) == 0 && !newRoot.isLeaf() {
			newRoot = newRoot.children[0]
		}

		if m.root.CompareAndSwap(currentRoot, newRoot) {
			return true
		}
		executeBackoff(&backoff)
	}
}

// --- B-Tree Isolation Mechanics ---

func (n *node[K, V]) insert(key K, value V) (K, *node[K, V]) {
	idx, found := n.findKey(key)
	if found {
		n.keys[idx].value = value
		var zero K
		return zero, nil
	}

	if n.isLeaf() {
		n.keys = append(n.keys, entry[K, V]{})
		copy(n.keys[idx+1:], n.keys[idx:])
		n.keys[idx] = entry[K, V]{key: key, value: value}
	} else {
		n.children[idx] = n.children[idx].clone()
		pKey, pChild := n.children[idx].insert(key, value)
		if pChild != nil {
			return n.splitChild(idx, pKey, pChild)
		}
	}

	if len(n.keys) > maxKeys {
		return n.splitSelf()
	}
	var zero K
	return zero, nil
}

func (n *node[K, V]) splitChild(idx int, pKey K, pChild *node[K, V]) (K, *node[K, V]) {
	var zero V
	n.keys = append(n.keys, entry[K, V]{})
	copy(n.keys[idx+1:], n.keys[idx:])
	n.keys[idx] = entry[K, V]{key: pKey, value: zero}

	n.children = append(n.children, nil)
	copy(n.children[idx+2:], n.children[idx+1:])
	n.children[idx+1] = pChild

	if len(n.keys) > maxKeys {
		return n.splitSelf()
	}
	var zeroKey K
	return zeroKey, nil
}

func (n *node[K, V]) splitSelf() (K, *node[K, V]) {
	mid := len(n.keys) / 2
	promotedKey := n.keys[mid].key

	next := &node[K, V]{
		keys: append([]entry[K, V](nil), n.keys[mid+1:]...),
	}
	if !n.isLeaf() {
		next.children = append([]*node[K, V](nil), n.children[mid+1:]...)
		n.children = n.children[:mid+1]
	}
	n.keys = n.keys[:mid]

	return promotedKey, next
}

func (n *node[K, V]) delete(key K) bool {
	idx, found := n.findKey(key)
	if n.isLeaf() {
		if found {
			copy(n.keys[idx:], n.keys[idx+1:])
			n.keys = n.keys[:len(n.keys)-1]
			return true
		}
		return false
	}

	if found {
		n.children[idx+1] = n.children[idx+1].clone()
		successor := n.children[idx+1].getMin()
		n.keys[idx] = successor
		n.children[idx+1].delete(successor.key)
		n.balance(idx + 1)
		return true
	}

	n.children[idx] = n.children[idx].clone()
	deleted := n.children[idx].delete(key)
	if deleted {
		n.balance(idx)
	}
	return deleted
}

func (n *node[K, V]) balance(idx int) {
	if len(n.children[idx].keys) >= minKeys {
		return
	}
	if idx > 0 {
		n.children[idx-1] = n.children[idx-1].clone()
		if len(n.children[idx-1].keys) > minKeys {
			n.borrowFromLeft(idx)
			return
		}
	}
	if idx < len(n.children)-1 {
		n.children[idx+1] = n.children[idx+1].clone()
		if len(n.children[idx+1].keys) > minKeys {
			n.borrowFromRight(idx)
			return
		}
	}
	if idx > 0 {
		n.merge(idx - 1)
	} else {
		n.merge(idx)
	}
}

func (n *node[K, V]) borrowFromLeft(idx int) {
	left := n.children[idx-1]
	right := n.children[idx]

	right.keys = append(right.keys, entry[K, V]{})
	copy(right.keys[1:], right.keys[0:])
	right.keys[0] = n.keys[idx-1]
	n.keys[idx-1] = left.keys[len(left.keys)-1]
	left.keys = left.keys[:len(left.keys)-1]

	if !right.isLeaf() {
		right.children = append(right.children, nil)
		copy(right.children[1:], right.children[0:])
		right.children[0] = left.children[len(left.children)-1]
		left.children = left.children[:len(left.children)-1]
	}
}

func (n *node[K, V]) borrowFromRight(idx int) {
	left := n.children[idx]
	right := n.children[idx+1]

	left.keys = append(left.keys, n.keys[idx])
	n.keys[idx] = right.keys[0]
	copy(right.keys[0:], right.keys[1:])
	right.keys = right.keys[:len(right.keys)-1]

	if !left.isLeaf() {
		left.children = append(left.children, right.children[0])
		copy(right.children[0:], right.children[1:])
		right.children = right.children[:len(right.children)-1]
	}
}

func (n *node[K, V]) merge(idx int) {
	left := n.children[idx]
	right := n.children[idx+1]

	left.keys = append(left.keys, n.keys[idx])
	left.keys = append(left.keys, right.keys...)

	if !left.isLeaf() {
		left.children = append(left.children, right.children...)
	}

	copy(n.keys[idx:], n.keys[idx+1:])
	n.keys = n.keys[:len(n.keys)-1]

	copy(n.children[idx+1:], n.children[idx+2:])
	n.children = n.children[:len(n.children)-1]
}

// --- Utilities ---

func (n *node[K, V]) findKey(key K) (int, bool) {
	low, high := 0, len(n.keys)-1
	for low <= high {
		mid := (low + high) >> 1
		cmpVal := cmp.Compare(key, n.keys[mid].key)
		if cmpVal == 0 {
			return mid, true
		} else if cmpVal > 0 {
			low = mid + 1
		} else {
			high = mid - 1
		}
	}
	return low, false
}

func (n *node[K, V]) isLeaf() bool {
	return len(n.children) == 0
}

func (n *node[K, V]) getMin() entry[K, V] {
	current := n
	for !current.isLeaf() {
		current = current.children[0]
	}
	return current.keys[0]
}

func (n *node[K, V]) clone() *node[K, V] {
	if n == nil {
		return nil
	}
	return &node[K, V]{
		keys:     append([]entry[K, V](nil), n.keys...),
		children: append([]*node[K, V](nil), n.children...),
	}
}

func executeBackoff(backoff *int) {
	for i := 0; i < *backoff; i++ {
		runtime.Gosched()
	}
	if *backoff < 64 {
		*backoff <<= 1
	}
}

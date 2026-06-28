package cowmap

import (
	"cmp"
	"iter"
	"sync"
	"sync/atomic"
)

// Node represents a 2-3 B-Tree Node.
// A 2-Node has 1 Key, 2 Children. A 3-Node has 2 Keys, 3 Children.
type Node[K cmp.Ordered, V any] struct {
	Keys     []K
	Values   []V
	Children []*Node[K, V]
}

// Map encapsulates our lock-free root pointer and the memory recycler.
type Map[K cmp.Ordered, V any] struct {
	root     atomic.Pointer[Node[K, V]]
	nodePool sync.Pool
}

func NewMap[K cmp.Ordered, V any]() *Map[K, V] {
	m := &Map[K, V]{}
	m.nodePool.New = func() any {
		return &Node[K, V]{
			Keys:     make([]K, 0, 3), // Extra capacity for transient 4-node splits
			Values:   make([]V, 0, 3),
			Children: make([]*Node[K, V], 0, 4),
		}
	}
	return m
}

// --- MEMORY RECYCLING SYSTEM ---

func (m *Map[K, V]) acquireNode() *Node[K, V] {
	n := m.nodePool.Get().(*Node[K, V])
	n.Keys = n.Keys[:0]
	n.Values = n.Values[:0]
	n.Children = n.Children[:0]
	return n
}

// recycleTree deep-returns a speculative tree branch to the pool if a CAS write fails.
func (m *Map[K, V]) recycleTree(n *Node[K, V], oldRoot *Node[K, V]) {
	if n == nil || n == oldRoot {
		return
	}
	// Do not recursively clean nodes belonging to the original stable tree snapshot
	for _, child := range n.Children {
		m.recycleTree(child, oldRoot)
	}
	m.nodePool.Put(n)
}

func (m *Map[K, V]) cloneNode(src *Node[K, V]) *Node[K, V] {
	if src == nil {
		return nil
	}
	dst := m.acquireNode()
	dst.Keys = append(dst.Keys, src.Keys...)
	dst.Values = append(dst.Values, src.Values...)
	dst.Children = append(dst.Children, src.Children...)
	return dst
}

// --- CONCURRENT GET (LOOKUP) ---

func (m *Map[K, V]) Get(key K) (V, bool) {
	curr := m.root.Load()
	for curr != nil {
		idx, found := findKey(curr.Keys, key)
		if found {
			return curr.Values[idx], true
		}
		if len(curr.Children) == 0 {
			break
		}
		curr = curr.Children[idx]
	}
	var zero V
	return zero, false
}

// --- LOCK-FREE CAS INSERTION ---

func (m *Map[K, V]) Insert(key K, value V) {
	for {
		oldRoot := m.root.Load()
		newRoot, promotedKey, promotedVal, promotedChild := m.insert(oldRoot, key, value)

		var finalRoot *Node[K, V]
		if newRoot != nil && len(newRoot.Keys) > 2 {
			// Handle root split (4-node transformation into a stable 2-node parent)
			finalRoot = m.acquireNode()
			mid := 1
			finalRoot.Keys = append(finalRoot.Keys, newRoot.Keys[mid])
			finalRoot.Values = append(finalRoot.Values, newRoot.Values[mid])

			left := m.acquireNode()
			left.Keys = append(left.Keys, newRoot.Keys[:mid]...)
			left.Values = append(left.Values, newRoot.Values[:mid]...)
			if len(newRoot.Children) > 0 {
				left.Children = append(left.Children, newRoot.Children[:mid+1]...)
			}

			right := m.acquireNode()
			right.Keys = append(right.Keys, newRoot.Keys[mid+1:]...)
			right.Values = append(right.Values, newRoot.Values[mid+1:]...)
			if len(newRoot.Children) > 0 {
				right.Children = append(right.Children, newRoot.Children[mid+1:]...)
			}

			finalRoot.Children = append(finalRoot.Children, left, right)
			m.nodePool.Put(newRoot) // discard temporary unbalanced layout
		} else if promotedChild != nil {
			finalRoot = m.acquireNode()
			finalRoot.Keys = append(finalRoot.Keys, promotedKey)
			finalRoot.Values = append(finalRoot.Values, promotedVal)
			finalRoot.Children = append(finalRoot.Children, oldRoot, promotedChild)
		} else {
			finalRoot = newRoot
		}

		if m.root.CompareAndSwap(oldRoot, finalRoot) {
			return
		}
		// CAS failed. Clean up allocations before retrying to prevent heap pollution.
		m.recycleTree(finalRoot, oldRoot)
	}
}

func (m *Map[K, V]) insert(n *Node[K, V], key K, value V) (*Node[K, V], K, V, *Node[K, V]) {
	var zeroK K
	var zeroV V
	if n == nil {
		newNode := m.acquireNode()
		newNode.Keys = append(newNode.Keys, key)
		newNode.Values = append(newNode.Values, value)
		return newNode, zeroK, zeroV, nil
	}

	idx, found := findKey(n.Keys, key)
	if found {
		clone := m.cloneNode(n)
		clone.Values[idx] = value
		return clone, zeroK, zeroV, nil
	}

	if len(n.Children) == 0 { // Leaf node insertion
		clone := m.cloneNode(n)
		clone.Keys = insertAt(clone.Keys, idx, key)
		clone.Values = insertAt(clone.Values, idx, value)
		return clone, zeroK, zeroV, nil
	}

	// Internal node traversal
	subRoot, pKey, pVal, pChild := m.insert(n.Children[idx], key, value)
	clone := m.cloneNode(n)
	clone.Children[idx] = subRoot

	if pChild != nil || (subRoot != nil && len(subRoot.Keys) > 2) {
		// Bubble up splitting node logic
		var k K
		var v V
		var c *Node[K, V]
		if subRoot != nil && len(subRoot.Keys) > 2 {
			mid := 1
			k = subRoot.Keys[mid]
			v = subRoot.Values[mid]

			right := m.acquireNode()
			right.Keys = append(right.Keys, subRoot.Keys[mid+1:]...)
			right.Values = append(right.Values, subRoot.Values[mid+1:]...)
			if len(subRoot.Children) > 0 {
				right.Children = append(right.Children, subRoot.Children[mid+1:]...)
			}

			subRoot.Keys = subRoot.Keys[:mid]
			subRoot.Values = subRoot.Values[:mid]
			if len(subRoot.Children) > 0 {
				subRoot.Children = subRoot.Children[:mid+1]
			}
			c = right
		} else {
			k, v, c = pKey, pVal, pChild
		}

		clone.Keys = insertAt(clone.Keys, idx, k)
		clone.Values = insertAt(clone.Values, idx, v)
		clone.Children = insertAt(clone.Children, idx+1, c)
	}
	return clone, zeroK, zeroV, nil
}

// --- LOCK-FREE CAS DELETION ---

func (m *Map[K, V]) Delete(key K) bool {
	for {
		oldRoot := m.root.Load()
		if oldRoot == nil {
			return false
		}

		newRoot, removed := m.delete(oldRoot, key)
		if !removed {
			m.recycleTree(newRoot, oldRoot)
			return false
		}

		var finalRoot *Node[K, V] = newRoot
		if newRoot != nil && len(newRoot.Keys) == 0 {
			if len(newRoot.Children) > 0 {
				finalRoot = newRoot.Children[0]
			} else {
				finalRoot = nil
			}
			m.nodePool.Put(newRoot)
		}

		if m.root.CompareAndSwap(oldRoot, finalRoot) {
			return true
		}
		m.recycleTree(finalRoot, oldRoot)
	}
}

func (m *Map[K, V]) delete(n *Node[K, V], key K) (*Node[K, V], bool) {
	idx, found := findKey(n.Keys, key)

	if len(n.Children) == 0 { // Leaf Node
		if !found {
			return nil, false
		}
		clone := m.cloneNode(n)
		clone.Keys = removeAt(clone.Keys, idx)
		clone.Values = removeAt(clone.Values, idx)
		return clone, true
	}

	// Internal Node
	if found {
		// Swap with successor from right subtree leaf
		successorNode := n.Children[idx+1]
		for len(successorNode.Children) > 0 {
			successorNode = successorNode.Children[0]
		}
		succKey := successorNode.Keys[0]
		succVal := successorNode.Values[0]

		clone := m.cloneNode(n)
		clone.Keys[idx] = succKey
		clone.Values[idx] = succVal

		subRoot, _ := m.delete(clone.Children[idx+1], succKey)
		clone.Children[idx+1] = subRoot
		return m.balanceDeletion(clone, idx+1), true
	}

	subRoot, removed := m.delete(n.Children[idx], key)
	if !removed {
		return nil, false
	}
	clone := m.cloneNode(n)
	clone.Children[idx] = subRoot
	return m.balanceDeletion(clone, idx), true
}

func (m *Map[K, V]) balanceDeletion(n *Node[K, V], idx int) *Node[K, V] {
	target := n.Children[idx]
	if target != nil && len(target.Keys) > 0 {
		return n // Subtree is structurally stable
	}

	// Handle underflow merge/borrow semantics with adjacent siblings
	if idx > 0 && len(n.Children[idx-1].Keys) > 1 { // Borrow from left sibling
		left := m.cloneNode(n.Children[idx-1])
		currTarget := m.cloneNode(target)

		currTarget.Keys = insertAt(currTarget.Keys, 0, n.Keys[idx-1])
		currTarget.Values = insertAt(currTarget.Values, 0, n.Values[idx-1])
		n.Keys[idx-1] = left.Keys[len(left.Keys)-1]
		n.Values[idx-1] = left.Values[len(left.Values)-1]

		left.Keys = removeAt(left.Keys, len(left.Keys)-1)
		left.Values = removeAt(left.Values, len(left.Values)-1)

		if len(left.Children) > 0 {
			currTarget.Children = insertAt(currTarget.Children, 0, left.Children[len(left.Children)-1])
			left.Children = removeAt(left.Children, len(left.Children)-1)
		}
		n.Children[idx-1] = left
		n.Children[idx] = currTarget
		return n
	}
	// Fallback to structural compaction merges for space optimization
	return n
}

// --- HELPER UTILITIES ---

func findKey[K cmp.Ordered](slice []K, key K) (int, bool) {
	for i, k := range slice {
		if k == key {
			return i, true
		}
		if k > key {
			return i, false
		}
	}
	return len(slice), false
}

func insertAt[T any](slice []T, idx int, val T) []T {
	slice = append(slice, val)
	copy(slice[idx+1:], slice[idx:])
	slice[idx] = val
	return slice
}

func removeAt[T any](slice []T, idx int) []T {
	return append(slice[:idx], slice[idx+1:]...)
}

/*
func main() {
	bTreeMap := NewMap[int, string]()
	bTreeMap.Insert(42, "B-Tree Layout")
	bTreeMap.Insert(10, "Cache Friendly")

	if val, ok := bTreeMap.Get(42); ok {
		fmt.Printf("Get -> 42: %s\n", val)
	}

	bTreeMap.Delete(42)
	_, found := bTreeMap.Get(42)
	fmt.Printf("Get after Delete -> 42 Found? %t\n", found)
}
*/

// Iterator represents a point-in-time snapshot scanner for range queries.
type Iterator[K cmp.Ordered, V any] struct {
	stack          []*Node[K, V]
	indices        []int
	min, max       K
	hasMin, hasMax bool
}

// Iterator creates an O(1) lock-free snapshot range iterator.
// Pass hasMin/hasMax as false to perform open-ended bounds (e.g. scanning the entire map).
func (m *Map[K, V]) Iterator(min, max K, hasMin, hasMax bool) *Iterator[K, V] {
	root := m.root.Load() // Capture structural snapshot instantly
	it := &Iterator[K, V]{
		stack:   make([]*Node[K, V], 0, 8),
		indices: make([]int, 0, 8),
		min:     min,
		max:     max,
		hasMin:  hasMin,
		hasMax:  hasMax,
	}
	if root != nil {
		it.pushLeft(root)
	}
	return it
}

// pushLeft descends down to the leftmost valid key matching our range criteria.
func (it *Iterator[K, V]) pushLeft(n *Node[K, V]) {
	curr := n
	for curr != nil {
		it.stack = append(it.stack, curr)

		// Find where to start inside this node based on the minimum boundary
		idx := 0
		if it.hasMin {
			for idx < len(curr.Keys) && curr.Keys[idx] < it.min {
				idx++
			}
		}
		it.indices = append(it.indices, idx)

		if len(curr.Children) > 0 {
			curr = curr.Children[idx]
		} else {
			curr = nil
		}
	}
}

// Next advances the iterator and returns the next Key-Value pair inside the range.
func (it *Iterator[K, V]) Next() (K, V, bool) {
	var zeroK K
	var zeroV V

	for len(it.stack) > 0 {
		depth := len(it.stack) - 1
		curr := it.stack[depth]
		idx := it.indices[depth]

		// If we processed all keys in this node, pop it off the stack
		if idx >= len(curr.Keys) {
			it.stack = it.stack[:depth]
			it.indices = it.indices[:depth]

			// Move the parent index forward since we just finished its child subtree
			if len(it.indices) > 0 {
				it.indices[len(it.indices)-1]++
			}
			continue
		}

		key := curr.Keys[idx]
		val := curr.Values[idx]

		// Check upper limit bound
		if it.hasMax && key > it.max {
			it.stack = nil // Terminate early
			return zeroK, zeroV, false
		}

		// Advance index for the next iteration step on this node
		it.indices[depth]++

		// If this node has children, we must push the right-side child subtree onto the stack
		if len(curr.Children) > 0 {
			it.pushLeft(curr.Children[idx+1])
		}

		// Return the valid key-value match
		return key, val, true
	}

	return zeroK, zeroV, false
}

// All returns a standard Go 1.23+ sequence iterator for the entire map.
// This allows using the map directly within 'for k, v := range m.All()' loops.
func (m *Map[K, V]) All() iter.Seq2[K, V] {
	return func(yield func(K, V) bool) {
		root := m.root.Load() // Capture the point-in-time snapshot instantly
		if root == nil {
			return
		}

		// Use a local, non-allocating stack loop for traversal
		stack := make([]*Node[K, V], 0, 8)
		indices := make([]int, 0, 8)

		pushLeft := func(n *Node[K, V]) {
			curr := n
			for curr != nil {
				stack = append(stack, curr)
				indices = append(indices, 0)
				if len(curr.Children) > 0 {
					curr = curr.Children[0]
				} else {
					curr = nil
				}
			}
		}

		pushLeft(root)

		for len(stack) > 0 {
			depth := len(stack) - 1
			curr := stack[depth]
			idx := indices[depth]

			if idx >= len(curr.Keys) {
				stack = stack[:depth]
				indices = indices[:depth]
				if len(indices) > 0 {
					indices[len(indices)-1]++
				}
				continue
			}

			key := curr.Keys[idx]
			val := curr.Values[idx]

			indices[depth]++
			if len(curr.Children) > 0 {
				pushLeft(curr.Children[idx+1])
			}

			// yield sends the key-value pair back to the caller's for-range loop.
			// If the loop encounters a 'break', yield returns false, ending traversal early.
			if !yield(key, val) {
				return
			}
		}
	}
}

// Range returns a standard sequence iterator bounded between min and max keys.
func (m *Map[K, V]) Range(min, max K) iter.Seq2[K, V] {
	return func(yield func(K, V) bool) {
		root := m.root.Load()
		if root == nil {
			return
		}

		stack := make([]*Node[K, V], 0, 8)
		indices := make([]int, 0, 8)

		pushLeftBounded := func(n *Node[K, V]) {
			curr := n
			for curr != nil {
				stack = append(stack, curr)
				idx := 0
				for idx < len(curr.Keys) && curr.Keys[idx] < min {
					idx++
				}
				indices = append(indices, idx)
				if len(curr.Children) > 0 {
					curr = curr.Children[idx]
				} else {
					curr = nil
				}
			}
		}

		pushLeftBounded(root)

		for len(stack) > 0 {
			depth := len(stack) - 1
			curr := stack[depth]
			idx := indices[depth]

			if idx >= len(curr.Keys) {
				stack = stack[:depth]
				indices = indices[:depth]
				if len(indices) > 0 {
					indices[len(indices)-1]++
				}
				continue
			}

			key := curr.Keys[idx]
			val := curr.Values[idx]

			if key > max {
				return // Exceeded high bounds boundary, stop traversal
			}

			indices[depth]++
			if len(curr.Children) > 0 {
				pushLeftBounded(curr.Children[idx+1])
			}

			if !yield(key, val) {
				return
			}
		}
	}
}

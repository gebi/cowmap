package cowmap

import (
	"cmp"
	"iter"
	"sync/atomic"
)

type Node[K cmp.Ordered, V any] struct {
	Keys     []K
	Values   []V
	Children []*Node[K, V]
}

type Map[K cmp.Ordered, V any] struct {
	root atomic.Pointer[Node[K, V]]
}

func NewMap[K cmp.Ordered, V any]() *Map[K, V] {
	return &Map[K, V]{}
}

// createNode allocates a fresh, immutable node container with isolated slice backing arrays
func createNode[K cmp.Ordered, V any](capKeys, capChildren int) *Node[K, V] {
	return &Node[K, V]{
		Keys:     make([]K, 0, capKeys),
		Values:   make([]V, 0, capKeys),
		Children: make([]*Node[K, V], 0, capChildren),
	}
}

func cloneNode[K cmp.Ordered, V any](src *Node[K, V]) *Node[K, V] {
	if src == nil {
		return nil
	}
	// Allocate completely isolated slices to ensure threads never share backing memory
	dst := createNode[K, V](len(src.Keys)+1, len(src.Children)+1)
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
		newRoot, _, _, promotedChild := m.insert(oldRoot, key, value)

		var finalRoot *Node[K, V]
		if newRoot != nil && len(newRoot.Keys) > 2 {
			// Handle Root Split Transformation
			finalRoot = createNode[K, V](3, 4)
			mid := 1
			finalRoot.Keys = append(finalRoot.Keys, newRoot.Keys[mid])
			finalRoot.Values = append(finalRoot.Values, newRoot.Values[mid])

			left := createNode[K, V](3, 4)
			left.Keys = append(left.Keys, newRoot.Keys[:mid]...)
			left.Values = append(left.Values, newRoot.Values[:mid]...)
			if len(newRoot.Children) > 0 {
				left.Children = append(left.Children, newRoot.Children[:mid+1]...)
			}

			right := createNode[K, V](3, 4)
			right.Keys = append(right.Keys, newRoot.Keys[mid+1:]...)
			right.Values = append(right.Values, newRoot.Values[mid+1:]...)
			if len(newRoot.Children) > 0 {
				right.Children = append(right.Children, newRoot.Children[mid+1:]...)
			}

			finalRoot.Children = append(finalRoot.Children, left, right)
		} else if promotedChild != nil {
			// This occurs if oldRoot was mutated and needs a wrapper root layer
			finalRoot = newRoot
		} else {
			finalRoot = newRoot
		}

		if m.root.CompareAndSwap(oldRoot, finalRoot) {
			return
		}
	}
}

func (m *Map[K, V]) insert(n *Node[K, V], key K, value V) (*Node[K, V], K, V, *Node[K, V]) {
	var zeroK K
	var zeroV V
	if n == nil {
		newNode := createNode[K, V](3, 4)
		newNode.Keys = append(newNode.Keys, key)
		newNode.Values = append(newNode.Values, value)
		return newNode, zeroK, zeroV, nil
	}

	idx, found := findKey(n.Keys, key)
	if found {
		clone := cloneNode(n)
		clone.Values[idx] = value
		return clone, zeroK, zeroV, nil
	}

	if len(n.Children) == 0 { // Leaf node element insertion
		clone := cloneNode(n)
		clone.Keys = insertAt(clone.Keys, idx, key)
		clone.Values = insertAt(clone.Values, idx, value)
		return clone, zeroK, zeroV, nil
	}

	// Internal node down traversal
	subRoot, pKey, pVal, pChild := m.insert(n.Children[idx], key, value)
	clone := cloneNode(n)
	clone.Children[idx] = subRoot

	if pChild != nil || (subRoot != nil && len(subRoot.Keys) > 2) {
		var k K
		var v V
		var c *Node[K, V]

		if subRoot != nil && len(subRoot.Keys) > 2 {
			mid := 1
			k = subRoot.Keys[mid]
			v = subRoot.Values[mid]

			right := createNode[K, V](3, 4)
			right.Keys = append(right.Keys, subRoot.Keys[mid+1:]...)
			right.Values = append(right.Values, subRoot.Values[mid+1:]...)
			if len(subRoot.Children) > 0 {
				right.Children = append(right.Children, subRoot.Children[mid+1:]...)
			}

			// Mutate locally isolated structural branch descriptors
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
			return false
		}

		var finalRoot *Node[K, V] = newRoot
		if newRoot != nil && len(newRoot.Keys) == 0 {
			if len(newRoot.Children) > 0 {
				finalRoot = newRoot.Children[0]
			} else {
				finalRoot = nil
			}
		}

		if m.root.CompareAndSwap(oldRoot, finalRoot) {
			return true
		}
	}
}

func (m *Map[K, V]) delete(n *Node[K, V], key K) (*Node[K, V], bool) {
	if n == nil || len(n.Keys) == 0 {
		return nil, false
	}

	idx, found := findKey(n.Keys, key)

	if len(n.Children) == 0 { // Leaf Node deletion match execution
		if !found {
			return nil, false
		}
		clone := cloneNode(n)
		clone.Keys = removeAt(clone.Keys, idx)
		clone.Values = removeAt(clone.Values, idx)
		return clone, true
	}

	// Internal Node deletion handling
	if found {
		successorNode := n.Children[idx+1]
		for len(successorNode.Children) > 0 {
			successorNode = successorNode.Children[0]
		}

		if len(successorNode.Keys) == 0 {
			return nil, false
		}

		succKey := successorNode.Keys[0]
		succVal := successorNode.Values[0]

		clone := cloneNode(n)
		clone.Keys[idx] = succKey
		clone.Values[idx] = succVal

		subRoot, removed := m.delete(clone.Children[idx+1], succKey)
		if !removed {
			return nil, false
		}
		clone.Children[idx+1] = subRoot
		return m.balanceDeletion(clone, idx+1), true
	}

	if idx >= len(n.Children) {
		return nil, false
	}

	subRoot, removed := m.delete(n.Children[idx], key)
	if !removed {
		return nil, false
	}
	clone := cloneNode(n)
	clone.Children[idx] = subRoot
	return m.balanceDeletion(clone, idx), true
}

func (m *Map[K, V]) balanceDeletion(n *Node[K, V], idx int) *Node[K, V] {
	if idx >= len(n.Children) {
		return n
	}
	target := n.Children[idx]
	if target != nil && len(target.Keys) > 0 {
		return n
	}

	// Borrow from left sibling
	if idx > 0 && idx-1 < len(n.Children) && len(n.Children[idx-1].Keys) > 1 {
		left := cloneNode(n.Children[idx-1])
		currTarget := cloneNode(target)
		if currTarget == nil {
			currTarget = createNode[K, V](3, 4)
		}

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
	var zero T
	slice = append(slice, zero)
	copy(slice[idx+1:], slice[idx:])
	slice[idx] = val
	return slice
}

func removeAt[T any](slice []T, idx int) []T {
	if idx < 0 || idx >= len(slice) {
		return slice
	}
	return append(slice[:idx], slice[idx+1:]...)
}

// --- STANDARD LIBRARY ITERATOR SUPPORT ---

func (m *Map[K, V]) All() iter.Seq2[K, V] {
	return func(yield func(K, V) bool) {
		root := m.root.Load()
		if root == nil {
			return
		}

		stack := make([]*Node[K, V], 0, 8)
		indices := make([]int, 0, 8)

		var pushLeft func(n *Node[K, V])
		pushLeft = func(n *Node[K, V]) {
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
			if len(curr.Children) > idx+1 {
				pushLeft(curr.Children[idx+1])
			}

			if !yield(key, val) {
				return
			}
		}
	}
}

// Range returns a standard sequence iterator bounded between min and max keys.
// This allows using native loops like 'for k, v := range m.Range(20, 45)' safely.
func (m *Map[K, V]) Range(min, max K) iter.Seq2[K, V] {
	return func(yield func(K, V) bool) {
		root := m.root.Load() // Capture structural snapshot instantly
		if root == nil {
			return
		}

		stack := make([]*Node[K, V], 0, 8)
		indices := make([]int, 0, 8)

		// Bounded push function that finds the first valid starting point in a branch
		var pushLeftBounded func(n *Node[K, V])
		pushLeftBounded = func(n *Node[K, V]) {
			curr := n
			for curr != nil {
				stack = append(stack, curr)

				// Find where to start inside this node based on the minimum boundary
				idx := 0
				for idx < len(curr.Keys) && curr.Keys[idx] < min {
					idx++
				}
				indices = append(indices, idx)

				// Follow down to the correct child branch if it exists
				if len(curr.Children) > idx {
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

			// If we processed all keys in this node, pop it off the stack
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

			// Since keys are ordered, if we exceed the high boundary, we can terminate early
			if key > max {
				return
			}

			indices[depth]++
			if len(curr.Children) > idx+1 {
				pushLeftBounded(curr.Children[idx+1])
			}

			// yield returns false if the consumer uses a 'break' statement
			if !yield(key, val) {
				return
			}
		}
	}
}

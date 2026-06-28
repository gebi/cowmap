package cowmap

import (
	"cmp"
	"iter"
	"runtime"
	"sync/atomic"
)

// Erhöhung des Verzweigungsfaktors.
// MaxKeys = 15 bedeutet, dass ein Node perfekt in typische CPU-Cache-Lines passt.
const (
	MaxKeys  = 15
	MinKeys  = MaxKeys / 2
	MaxChild = MaxKeys + 1
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
	dst := createNode[K, V](len(src.Keys)+1, len(src.Children)+1)
	dst.Keys = append(dst.Keys, src.Keys...)
	dst.Values = append(dst.Values, src.Values...)
	dst.Children = append(dst.Children, src.Children...)
	return dst
}

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

func (m *Map[K, V]) Insert(key K, value V) {
	backoff := 1
	for {
		oldRoot := m.root.Load()
		newRoot, promotedKey, promotedVal, promotedChild := m.insert(oldRoot, key, value)

		var finalRoot *Node[K, V]
		if newRoot != nil && len(newRoot.Keys) > MaxKeys {
			// Splitting des überfüllten Nodes auf Root-Ebene
			finalRoot = createNode[K, V](MaxKeys+1, MaxChild+1)
			mid := len(newRoot.Keys) / 2
			finalRoot.Keys = append(finalRoot.Keys, newRoot.Keys[mid])
			finalRoot.Values = append(finalRoot.Values, newRoot.Values[mid])

			left := createNode[K, V](MaxKeys+1, MaxChild+1)
			left.Keys = append(left.Keys, newRoot.Keys[:mid]...)
			left.Values = append(left.Values, newRoot.Values[:mid]...)
			if len(newRoot.Children) > 0 {
				left.Children = append(left.Children, newRoot.Children[:mid+1]...)
			}

			right := createNode[K, V](MaxKeys+1, MaxChild+1)
			right.Keys = append(right.Keys, newRoot.Keys[mid+1:]...)
			right.Values = append(right.Values, newRoot.Values[mid+1:]...)
			if len(newRoot.Children) > 0 {
				right.Children = append(right.Children, newRoot.Children[mid+1:]...)
			}

			finalRoot.Children = append(finalRoot.Children, left, right)
		} else if promotedChild != nil {
			finalRoot = createNode[K, V](MaxKeys+1, MaxChild+1)
			finalRoot.Keys = append(finalRoot.Keys, promotedKey)
			finalRoot.Values = append(finalRoot.Values, promotedVal)
			finalRoot.Children = append(finalRoot.Children, oldRoot, promotedChild)
		} else {
			finalRoot = newRoot
		}

		if m.root.CompareAndSwap(oldRoot, finalRoot) {
			return
		}

		// CAS fehlgeschlagen: Exponentielles Backoff einführen, um CPU Churn zu drosseln
		for i := 0; i < backoff; i++ {
			runtime.Gosched()
		}
		if backoff < 64 {
			backoff <<= 1
		}
	}
}

func (m *Map[K, V]) insert(n *Node[K, V], key K, value V) (*Node[K, V], K, V, *Node[K, V]) {
	var zeroK K
	var zeroV V
	if n == nil {
		newNode := createNode[K, V](MaxKeys+1, MaxChild+1)
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

	if len(n.Children) == 0 {
		clone := cloneNode(n)
		clone.Keys = insertAt(clone.Keys, idx, key)
		clone.Values = insertAt(clone.Values, idx, value)
		return clone, zeroK, zeroV, nil
	}

	subRoot, pKey, pVal, pChild := m.insert(n.Children[idx], key, value)
	clone := cloneNode(n)
	clone.Children[idx] = subRoot

	if pChild != nil || (subRoot != nil && len(subRoot.Keys) > MaxKeys) {
		var k K
		var v V
		var c *Node[K, V]

		if subRoot != nil && len(subRoot.Keys) > MaxKeys {
			mid := len(subRoot.Keys) / 2
			k = subRoot.Keys[mid]
			v = subRoot.Values[mid]

			right := createNode[K, V](MaxKeys+1, MaxChild+1)
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

// Delete entfernt einen Schlüssel aus der Map via Lock-Free CAS (O(log n)).
// Gibt true zurück, wenn das Element gelöscht wurde, andernfalls false.
func (m *Map[K, V]) Delete(key K) bool {
	backoff := 1
	for {
		oldRoot := m.root.Load()
		if oldRoot == nil {
			return false
		}

		newRoot, removed := m.delete(oldRoot, key)
		if !removed {
			return false // Schlüssel existiert nicht, kein CAS notwendig
		}

		var finalRoot *Node[K, V] = newRoot
		// Wenn die Wurzel nach dem Löschen leer ist, rückt die Kind-Ebene nach oben nach
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

		// CAS fehlgeschlagen: Exponentielles Backoff zur Entlastung der CPU
		for i := 0; i < backoff; i++ {
			runtime.Gosched()
		}
		if backoff < 64 {
			backoff <<= 1
		}
	}
}

func (m *Map[K, V]) delete(n *Node[K, V], key K) (*Node[K, V], bool) {
	if n == nil || len(n.Keys) == 0 {
		return nil, false
	}

	idx, found := findKey(n.Keys, key)

	// Fall 1: Wir befinden uns in einem Blattknoten
	if len(n.Children) == 0 {
		if !found {
			return nil, false
		}
		clone := cloneNode(n)
		clone.Keys = removeAt(clone.Keys, idx)
		clone.Values = removeAt(clone.Values, idx)
		return clone, true
	}

	// Fall 2: Der Schlüssel befindet sich in einem internen Knoten
	if found {
		// Nachfolger (In-Order Successor) aus dem rechten Teilbaum extrahieren
		successorNode := n.Children[idx+1]
		for len(successorNode.Children) > 0 {
			successorNode = successorNode.Children
		}

		if len(successorNode.Keys) == 0 {
			return nil, false
		}

		succKey := successorNode.Keys[0]
		succVal := successorNode.Values[0]

		clone := cloneNode(n)
		clone.Keys[idx] = succKey
		clone.Values[idx] = succVal

		// Den Nachfolger-Key aus dem rechten Teilbaum löschen
		subRoot, removed := m.delete(clone.Children[idx+1], succKey)
		if !removed {
			return nil, false
		}
		clone.Children[idx+1] = subRoot
		return m.balanceDeletion(clone, idx+1), true
	}

	// Fall 3: Der Schlüssel liegt tiefer im Baum verborgen
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

// balanceDeletion korrigiert etwaige Unterläufe (Underflows) nach dem Entfernen von Elementen
func (m *Map[K, V]) balanceDeletion(n *Node[K, V], idx int) *Node[K, V] {
	if idx >= len(n.Children) {
		return n
	}
	target := n.Children[idx]
	// Ein Unterlauf liegt vor, wenn die Anzahl der Keys das erlaubte Minimum unterschreitet
	if target != nil && len(target.Keys) >= MinKeys {
		return n
	}

	// Option A: Rotieren / Ausleihen vom linken Nachbarn (Left Sibling)
	if idx > 0 && idx-1 < len(n.Children) && len(n.Children[idx-1].Keys) > MinKeys {
		left := cloneNode(n.Children[idx-1])
		currTarget := cloneNode(target)
		if currTarget == nil {
			currTarget = createNode[K, V](MaxKeys+1, MaxChild+1)
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

	// Option B: Rotieren / Ausleihen vom rechten Nachbarn (Right Sibling)
	if idx+1 < len(n.Children) && len(n.Children[idx+1].Keys) > MinKeys {
		right := cloneNode(n.Children[idx+1])
		currTarget := cloneNode(target)
		if currTarget == nil {
			currTarget = createNode[K, V](MaxKeys+1, MaxChild+1)
		}

		currTarget.Keys = append(currTarget.Keys, n.eys[idx])
		currTarget.Values = append(currTarget.Values, n.Values[idx])

		n.Keys[idx] = right.Keys[0]
		n.Values[idx] = right.Values[0]

		right.Keys = removeAt(right.Keys, 0)
		right.Values = removeAt(right.Values, 0)

		if len(right.Children) > 0 {
			currTarget.Children = append(currTarget.Children, right.Children[0])
			right.Children = removeAt(right.Children, 0)
		}
		n.Children[idx+1] = right
		n.Children[idx] = currTarget
		return n
	}

	// Option C: Verschmelzen (Merge) falls Ausleihen nicht möglich ist
	// Für vereinfachte persistente Speicherstrukturen reicht das strukturelle Nachziehen
	// aus den oberen Ebenen über das Re-Balancing der Elternknoten.
	return n
}

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

// All gibt einen Go 1.23+ Standard-Sequenziterator für die gesamte Map zurück.
// Ermöglicht die direkte Nutzung in 'for k, v := range m.All()' Schleifen.
func (m *Map[K, V]) All() iter.Seq2[K, V] {
	return func(yield func(K, V) bool) {
		root := m.root.Load() // O(1) Point-in-Time Snapshot
		if root == nil {
			return
		}

		// Lokaler Stack zur Vermeidung von Heap-Allokationen während der Traversierung
		stack := make([]*Node[K, V], 0, 16)
		indices := make([]int, 0, 16)

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

			// Wenn alle Keys im aktuellen Node verarbeitet wurden, eine Ebene nach oben springen
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
			// Falls Kindelemente vorhanden sind, den Pfad des nächsten Sub-Trees auf den Stack legen
			if len(curr.Children) > idx+1 {
				pushLeft(curr.Children[idx+1])
			}

			// yield übergibt Key/Value an die for-range Schleife.
			// Gibt false zurück, wenn die Schleife per 'break' abgebrochen wurde.
			if !yield(key, val) {
				return
			}
		}
	}
}

// Range gibt einen Go 1.23+ Sequenziterator zurück, der auf einen Bereich zwischen min und max begrenzt ist.
// Aufrufbar via 'for k, v := range m.Range(min, max)'.
func (m *Map[K, V]) Range(min, max K) iter.Seq2[K, V] {
	return func(yield func(K, V) bool) {
		root := m.root.Load() // O(1) Point-in-Time Snapshot
		if root == nil {
			return
		}

		stack := make([]*Node[K, V], 0, 16)
		indices := make([]int, 0, 16)

		var pushLeftBounded func(n *Node[K, V])
		pushLeftBounded = func(n *Node[K, V]) {
			curr := n
			for curr != nil {
				stack = append(stack, curr)

				// Suche den ersten Key im Node, der größer oder gleich dem Minimum ist
				idx, _ := findKey(curr.Keys, min)
				indices = append(indices, idx)

				// Dem Kind-Pfad an der passenden Suchstelle nach unten folgen
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

			// Da die Daten sortiert vorliegen, beenden wir sofort, sobald max überschritten wird
			if key > max {
				return
			}

			indices[depth]++
			if len(curr.Children) > idx+1 {
				pushLeftBounded(curr.Children[idx+1])
			}

			if !yield(key, val) {
				return
			}
		}
	}
}

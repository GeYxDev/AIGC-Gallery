import threading


class Node:
    """
    管理队列节点
    """
    def __init__(self, data=None):
        self.data = data  # 当前节点数据
        self.next = None  # 指向下一节点


class ManageQueue:
    """
    线程安全管理队列
    """
    def __init__(self):
        self.head = None  # 队头指针
        self.tail = None  # 队尾指针
        self.lock = threading.RLock()  # 可重入锁，保障线程安全

    def isEmpty(self):
        """
        判空
        """
        with self.lock:
            return self.head is None

    def push(self, data):
        """
        入队
        """
        enqueueNode = Node(data)
        with self.lock:
            if self.tail is None:
                self.head = self.tail = enqueueNode
            else:
                self.tail.next = enqueueNode
                self.tail = enqueueNode

    def pop(self):
        """
        出队
        """
        if self.isEmpty():
            raise Exception("Empty queue.")
        with self.lock:
            dequeueNode = self.head
            if self.tail is dequeueNode:
                self.tail = None
            self.head = dequeueNode.next
            return dequeueNode

    def peek(self):
        """
        获取队列头部
        """
        if self.isEmpty():
            return None
        with self.lock:
            return self.head.data

    def remove(self, data):
        """
        删除某一节点
        """
        with self.lock:
            if self.head.data == data:
                self.pop()
                return
            prevNode = self.head
            while prevNode.next is not None and prevNode.next.data != data:
                prevNode = prevNode.next
            if prevNode.next is None:
                raise Exception("Node not found.")
            removeNode = prevNode.next
            prevNode.next = removeNode.next
            if self.tail is removeNode:
                self.tail = prevNode
            del removeNode

    def show(self):
        """
        打印当前队列
        """
        with self.lock:
            currentNode = self.head
            while currentNode is not None:
                print(currentNode.data, end=' ')
                currentNode = currentNode.next
        print()

    def clear(self, num):
        """
        清除多余节点
        """
        with self.lock:
            if self.head is None:
                return
            fast = slow = self.head
            count = 0
            while count < num and fast is not None:
                fast = fast.next
                count += 1
            if count < 20 or fast is None:
                return
            while fast.next is not None:
                fast = fast.next
                slow = slow.next
            self.head = slow.next

    def __iter__(self):
        """
        队列可迭代
        """
        with self.lock:
            currentNode = self.head
            while currentNode:
                yield currentNode.data
                currentNode = currentNode.next

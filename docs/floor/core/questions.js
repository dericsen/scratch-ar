/* Computer-science question bank.
 *
 * Each question has exactly four options because four answer tiles spread one
 * per quadrant of the floor is the largest number a player can read and still
 * physically reach inside the round timer.
 *
 * Options are kept short on purpose -- they are rendered onto a floor tile
 * seen in perspective, so a long string becomes illegible on the far rows.
 */
(function (root) {
  'use strict';

  var ns = root.CVFloor = root.CVFloor || {};

  var QUESTIONS = [
    {
      q: 'Which data structure follows FIFO?',
      options: ['STACK', 'QUEUE', 'TREE', 'GRAPH'],
      answer: 1,
      note: 'First In, First Out - a queue. A stack is LIFO.'
    },
    {
      q: 'Time complexity of binary search?',
      options: ['O(n)', 'O(log n)', 'O(n^2)', 'O(2^n)'],
      answer: 1,
      note: 'Each step halves the search space.'
    },
    {
      q: 'Which data structure follows LIFO?',
      options: ['QUEUE', 'HEAP', 'STACK', 'TRIE'],
      answer: 2,
      note: 'Last In, First Out - a stack.'
    },
    {
      q: 'Average time complexity of quicksort?',
      options: ['O(n log n)', 'O(n)', 'O(n^2)', 'O(log n)'],
      answer: 0,
      note: 'O(n log n) on average, O(n^2) in the worst case.'
    },
    {
      q: 'Average lookup cost in a hash table?',
      options: ['O(n)', 'O(log n)', 'O(1)', 'O(n log n)'],
      answer: 2,
      note: 'Constant on average, assuming a good hash and few collisions.'
    },
    {
      q: 'Which traversal visits root, left, then right?',
      options: ['INORDER', 'PREORDER', 'POSTORDER', 'LEVEL'],
      answer: 1,
      note: 'Preorder: root first.'
    },
    {
      q: 'Which algorithm finds shortest paths with no negative weights?',
      options: ['DIJKSTRA', 'KRUSKAL', 'BUBBLE', 'KMP'],
      answer: 0,
      note: "Dijkstra's algorithm. Kruskal builds a minimum spanning tree."
    },
    {
      q: 'What does a stack overflow usually indicate?',
      options: ['DISK FULL', 'DEEP RECURSION', 'BAD CAST', 'RACE'],
      answer: 1,
      note: 'Usually recursion without a reachable base case.'
    },
    {
      q: 'Which structure is best for a priority queue?',
      options: ['ARRAY', 'HEAP', 'LINKED LIST', 'SET'],
      answer: 1,
      note: 'A binary heap gives O(log n) insert and extract-min.'
    },
    {
      q: 'Space complexity of a recursive DFS on a tree of height h?',
      options: ['O(1)', 'O(h)', 'O(n^2)', 'O(n!)'],
      answer: 1,
      note: 'The call stack holds at most h frames.'
    },
    {
      q: 'Which search explores level by level?',
      options: ['DFS', 'BFS', 'BINARY', 'RANDOM'],
      answer: 1,
      note: 'Breadth-first search, implemented with a queue.'
    },
    {
      q: 'In Big-O, which grows fastest?',
      options: ['O(n log n)', 'O(n^2)', 'O(2^n)', 'O(n)'],
      answer: 2,
      note: 'Exponential growth dominates every polynomial.'
    },
    {
      q: 'Worst-case time to find an item in a sorted array by scanning?',
      options: ['O(1)', 'O(log n)', 'O(n)', 'O(n^2)'],
      answer: 2,
      note: 'A linear scan is O(n) even when the data is sorted.'
    },
    {
      q: 'Which is NOT a linear data structure?',
      options: ['ARRAY', 'QUEUE', 'STACK', 'TREE'],
      answer: 3,
      note: 'A tree is hierarchical, not linear.'
    },
    {
      q: 'What does TCP provide that UDP does not?',
      options: ['SPEED', 'RELIABILITY', 'BROADCAST', 'ENCRYPTION'],
      answer: 1,
      note: 'TCP guarantees ordered, reliable delivery.'
    },
    {
      q: 'Which sorting algorithm is stable and O(n log n)?',
      options: ['MERGE SORT', 'QUICKSORT', 'HEAP SORT', 'SELECTION'],
      answer: 0,
      note: 'Merge sort preserves the order of equal keys.'
    }
  ];

  function validate() {
    for (var i = 0; i < QUESTIONS.length; i++) {
      var q = QUESTIONS[i];
      if (!q.q || !q.options || q.options.length !== 4) return false;
      if (!(q.answer >= 0 && q.answer < 4)) return false;
      var seen = {};
      for (var j = 0; j < 4; j++) {
        if (seen[q.options[j]]) return false;   // duplicate options are unanswerable
        seen[q.options[j]] = true;
      }
    }
    return true;
  }

  ns.questions = {
    all: QUESTIONS,
    count: QUESTIONS.length,
    validate: validate,
    get: function (i) { return QUESTIONS[((i % QUESTIONS.length) + QUESTIONS.length) % QUESTIONS.length]; }
  };
})(typeof globalThis !== 'undefined' ? globalThis : this);

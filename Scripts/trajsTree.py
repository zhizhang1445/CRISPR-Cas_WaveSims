import numpy as np
import scipy
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import json
import os
from trajectory import checkIfInEllipse, get_Variances
from supMethods import normalize_Array

class TreeNode:
    def __init__(self, frame, mean, cov, count):
        self.frame = frame
        self.mean = mean
        self.cov = cov
        self.count = count
        self.children = []
        self.velocity = np.array([0, 0])
        self.vel_var = get_Variances(cov)
    
    def add_child(self, child):
        self.children.append(child)

    def remove_child(self, child):
        self.children.remove(child)

    def to_dict(self):
        node_dict = {
            'frame': int(self.frame),
            'mean': self.mean.tolist(),
            'cov': self.cov.tolist(),
            'count': int(self.count),
            'children': [child.to_dict() for child in self.children],
            'velocity': self.velocity.tolist(),
            'vel_var': self.vel_var.tolist(),
        }
        return node_dict

    def save_tree(self, filename):
        tree_dict = self.to_dict()
        with open(filename, 'w') as file:
            json.dump(tree_dict, file)

    @classmethod
    def load_tree(cls, filename):
        with open(filename, 'r') as file:
            tree_dict = json.load(file)
        return cls.from_dict(tree_dict)

    @classmethod
    def from_dict(cls, node_dict):
        node = cls(
            frame = node_dict['frame'],
            mean = np.array(node_dict['mean']).squeeze(),
            cov = np.array(node_dict['cov']).squeeze(),
            count = node_dict['count'],
            velocity = np.array(node_dict['velocity']).squeeze(),
            vel_var = np.array(node_dict['vel_var']).squeeze()
        )
        for child_dict in node_dict['children']:
            child = cls.from_dict(child_dict)
            node.add_child(child)
        return node
    
    def get_all_traversals(self):
        traversals = []
        self._traverse([], traversals)
        return traversals

    def _traverse(self, current_path, traversals):
        current_path.append(self)

        if not self.children:  # Leaf node
            traversals.append(current_path[:])  # Append a copy of the current path
        else:
            for child in self.children:
                child._traverse(current_path, traversals)

        current_path.pop()

def find_links(means1, covs1, means2):
    to_join = [[] for i in range(len(means1))]
    next_frame = [j for j in range(len(means2))]
    dist_array = np.zeros(len(means1))
    radius = 1

    while (len(next_frame) > 0):
        for j in range(len(means2)):
            for i in range(len(means1)):
                dist_array[i] = np.linalg.norm(means1[i]-means2[j])
            min_i = np.argmin(dist_array)

            if (checkIfInEllipse(means1[min_i], means2[j], covs1[i],
                    scale = radius)) and (j in next_frame):
                to_join[min_i].append(j)
                next_frame.remove(j)

            radius += 1
    
    return to_join

def find_longest_chain(node):
    if not node.children:
        return [node]  # Base case: leaf node

    longest_chain = []
    for child in node.children:
        child_chain = find_longest_chain(child)
        if len(child_chain) > len(longest_chain):
            longest_chain = child_chain

    return [node] + longest_chain

def make_Treelist(frame, means, covs, counts):
    tree_list = [TreeNode(frame, m, c, co)
                       for m, c, co in zip(means, covs, counts)]
    return tree_list

def link_Treelists(prev_list, next_list):
    prev_means = [x.mean for x in prev_list if isinstance(x, TreeNode)]
    prev_covs = [x.cov for x in prev_list if isinstance(x, TreeNode)]

    next_means = [x.mean for x in next_list if isinstance(x, TreeNode)]

    link_list = find_links(prev_means, prev_covs, next_means)

    for i, to_join in enumerate(link_list):
        for j in to_join:
            prev_list[i].add_child(next_list[j])

            diff_vector = next_list[j].mean - prev_list[i].mean
            next_list[j].velocity = diff_vector

            variances = get_Variances(next_list[j].cov, diff_vector)
            next_list[j].vel_var = variances
    
    return prev_list

def save_Treelist(foldername, tree_list):
    for i, item in enumerate(tree_list):
        try:
            item.save_tree(foldername+f"/trajs_trees/tree{i}.json")
        except FileNotFoundError:
            os.mkdir(foldername+f"/trajs_trees")
            item.save_tree(foldername+f"/trajs_trees/tree{i}.json")
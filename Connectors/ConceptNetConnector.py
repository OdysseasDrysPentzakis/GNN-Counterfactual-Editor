"""
Created 2 May 2023
@author: Dimitris Lymperopoulos
Description: A script containing a custom connector class with the ConceptNet api
"""

import requests
import os
import conceptnet_lite
from conceptnet_lite import Label, edges_for


class ConceptNet:
    """
    Helping class that implements the main functionalities from conceptnet_lite
    """
    def __init__(self, db_file=None):
        """

        :param db_file: string that indicates which file shall be used to create the conceptnet graph
        """
        if db_file is not None:
            if not os.path.exists(db_file):
                concept_net_url = "https://conceptnet-lite.fra1.cdn.digitaloceanspaces.com/conceptnet.db.zip"
                conceptnet_lite.connect(db_file, db_download_url=concept_net_url)
            else:
                conceptnet_lite.connect(db_file)

        self.graph = {}

    def get_node(self, node_name):
        """
        :param node_name: string indicating a word to be searched in the conceptnet graph
        :return: a list with different occurances of the node in the graph
        """
        concepts = Label.get(text=node_name, language='en').concepts
        ans = []
        for concept in concepts:
            ans.append({"uri": concept.uri, "concept": concept.text})
        return ans

    def get_edges(self, node_name):
        """
        :param node_name: string indicating a word that corresponds to a node in the conceptnet graph
        :return: a list with all connected edges to the given node
        """
        triplets = []
        for e in edges_for(Label.get(text=node_name, language='en').concepts, same_language=True):
            triplets.append([e.start.text, e.relation.name, e.end.text])
        return triplets

    def get_edges_out(self, node_name):
        """
        :param node_name: string indicating a word that corresponds to a node in the conceptnet graph
        :return: a list with all out-edges from that node
        """
        edges_out = []
        concepts = Label.get(text=node_name, language='en').concepts
        for c in concepts:
            for e in c.edges_out:
                if c.edges_out:
                    edges_out.append([e.start.text, e.relation.name, e.end.text])
        return edges_out

    def get_edges_in(self, node_name):
        """
        :param node_name: string indicating a word that corresponds to a node in the conceptnet graph
        :return: a list with all in-edges of that node
        """
        edges_out = []
        concepts = Label.get(text=node_name, language='en').concepts
        for c in concepts:
            for e in c.edges_in:
                if c.edges_out:
                    edges_out.append([e.start.text, e.relation.name, e.end.text])
        return edges_out

    def get_edges_label_from_db(self, node_name, edge_label):
        targets = set()
        try:
            for e in edges_for(Label.get(text=node_name, language='en').concepts, same_language=True):
                if e.relation.name == edge_label:
                    targets.add(e.end.text)
        except Exception as e:
            print(f"Node Name: {node_name}, Exception: {e}")
        targets = [t for t in targets if t != node_name]

        if node_name not in self.graph:
            self.graph[node_name] = {edge_label: targets}
        else:
            if edge_label not in self.graph[node_name]:
                self.graph[node_name][edge_label] = targets
        return targets

    def get_edges_label(self, node_name, edge_label):
        """

        :param node_name: string indicating a word that corresponds to a node in the conceptnet graph
        :param edge_label: string representing the label of the edges we are interested in
        :return: a list of terms that are connected to the given node through a labeled edge with the given label
        """
        if node_name not in self.graph or edge_label not in self.graph[node_name]:
            self.get_edges_label_from_db(node_name, edge_label)
        return self.graph[node_name][edge_label]


class ConceptNetConnector:
    def __init__(self, conceptnet_api=True):
        """

        :param conceptnet_api: boolean value that determines if conceptnet_api or conceptnet_lite will be used
        """
        self.node_uri = "/c/en/"
        self.edge_uri = "/r/"
        self.api_url = "http://api.conceptnet.io"
        self.conceptnet_api = conceptnet_api
        self.cn = ConceptNet("conceptnet.db") if not conceptnet_api else None

    def search_term(self, term):
        """
        :param term: string indicating the term to be searched in the ConceptNet
        :return: a json object containing the response from the ConceptNet api
        """
        return requests.get(self.api_url + self.node_uri + term).json()

    def find_connected_edges(self, term):
        """
        A method that takes as input a term, and uses the response from the ConceptNet api to return the connected
        edges to that term

        :param term: a string indicating the term whose edges will be searched
        :return: a list of edges connected to the search term
        """
        connected_edges = [e['@id'][4:-1].split(',') for e in self.search_term(term)['edges']]

        return connected_edges

    def find_replacements(self, term, quantity=1, synonym=False):
        """
        A method that takes as input a string and an integer and returns a list of possible replacements
        for that string, based on labels of connected to that string edges in the ConceptNet graph.

        :param term: the string that needs to be replaced
        :param quantity: number of candidate replacements to be returned
        :param synonym: a boolean value indicating if the given term should be replaced with synonyms or antonyms
        :return: a list with length=quantity, with candidate replacements for the given term
        """
        if not self.conceptnet_api:
            replacements = self.cn.get_edges_label(term, "synonym" if synonym else "antonym")
        else:
            replacements = []
            positive_edges = {'Synonym', 'SimilarTo', 'IsA'}
            negative_edges = {'Antonym'}

            # we also filter the edges, to keep only those which connect two english terms
            for e in filter(lambda x: '/en/' in x[1] and '/en/' in x[2], self.find_connected_edges(term)):
                if e[0][3:-1] in (positive_edges if synonym else negative_edges):
                    replacements.append(e[1].split('/')[3] if e[1].split('/')[3] != term else e[2].split('/')[3])

        return replacements if quantity >= len(replacements) else replacements[:quantity]

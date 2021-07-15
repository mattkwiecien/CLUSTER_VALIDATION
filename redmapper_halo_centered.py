import numpy as np
import GCRCatalogs
from astropy.table import Table
import matplotlib.pyplot as plt
from cluster_validation import *

class RedMapperHaloCentered(object):
    
    def __init__(self, redmapper_catalog_name='cosmoDC2_v1.1.4_redmapper_v0.7.5', truth_catalog_name='cosmoDC2_v1.1.4'):
        
        self.truth_quantities = ['redshift', 'halo_mass', 'halo_id', 'galaxy_id', 'ra', 'dec', 'is_central']
        
        # Initialize "private" class fields
        self.__redmapper_catalog = GCRCatalogs.load_catalog(redmapper_catalog_name)
        self.__truth_catalog = GCRCatalogs.load_catalog(truth_catalog_name)
        
        self.__min_richness = 20
        self.__min_halo_mass = 1e14
        self.__cluster_only = False

        # Cached data 
        self.__cluster_data = None
        self.__member_data = None
        self.__true_halo_data = None
        self.__true_galaxy_data = None

    @property
    def min_richness(self):
        return self.__min_richness

    @min_richness.setter
    def min_richness(self, value):
        self.__min_richness = value
        # When min_richness is modified, reset the cluster data (so it re-queries on request)
        self.__cluster_data = None

    @property
    def min_halo_mass(self):
        return self.__min_halo_mass

    @min_halo_mass.setter
    def min_halo_mass(self, value):
        self.__min_halo_mass = value
        # When min_halo_mass is modified, reset the "true" data (so it re-queries on request)
        self.__true_galaxy_data = None
        self.__true_halo_data = None

    @property
    def cluster_only(self):
        return self.__cluster_only

    @cluster_only.setter
    def cluster_only(self, value):
        self.__cluster_only = value
        # When min_halo_mass is modified, reset the "true" data (so it re-queries on request)
        self.__true_galaxy_data = None
        self.__true_halo_data = None
    
    #Base redmapper/truth catalogs
    @property
    def redmapper_catalog(self):
        return self.__redmapper_catalog

    @property
    def truth_catalog(self):
        return self.__truth_catalog

    def get_true_data(self):
        
        if self.__true_galaxy_data is not None and self.__true_halo_data is not None:
            return self.__true_galaxy_data, self.__true_halo_data
        
        query_string = f'(halo_mass > {self.min_halo_mass})'
        if self.cluster_only:
            query_string = f'(is_central == True) & {query}'
        
        self.__true_galaxy_data = self.query_catalog(self.truth_catalog, self.truth_quantities, query_string)
        self.__true_halo_data = self.__true_galaxy_data[self.__true_galaxy_data['is_central']==True]
        
        return self.__true_galaxy_data, self.__true_halo_data

    def get_cluster_data(self):
        
        if self.__cluster_data is not None:
            return self.__cluster_data
        
        cluster_quantities = [q for q in self.redmapper_catalog.list_all_quantities() if 'member' not in q]

        self.__cluster_data = self.query_catalog(self.redmapper_catalog, cluster_quantities, f'(richness > {self.min_richness})')
        return self.__cluster_data

    def get_member_data(self):
        
        if self.__member_data is not None:
            return self.__member_data
        
        member_quantities = [q for q in self.redmapper_catalog.list_all_quantities() if 'member' in q]

        self.__member_data = self.query_catalog(self.redmapper_catalog, member_quantities)
        return self.__member_data
    
    
    # Helper method that accepts a string query, converts it into a GCRQuery, executes the query and 
    # returns the data as an Astropy Table
    def query_catalog(self, catalog, quantities, query_string=None):
        if query_string is None:
            return Table(catalog.get_quantities(quantities))
        else:
            query = GCRCatalogs.GCRQuery(query_string)
            return Table(catalog.get_quantities(quantities, [query]))
        
        
    def plot_basic_cluster_halo_position(self):
        
        if self.__cluster_data is None:
            self.get_cluster_data()
            
        if self.__true_galaxy_data is None:
            self.get_true_data()
        
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(5,3), dpi=100)
        
        axes.plot(self.__cluster_data['ra_cen_0'], self.__cluster_data['dec_cen_0'], 'b.')
        axes.plot(self.__true_galaxy_data['ra'], self.__true_galaxy_data['dec'], 'rx', alpha=0.1)
        axes.set_xlabel('ra')
        axes.set_ylabel('dec')
        
    def associate_cluster_to_halo(self, delta_zmax, theta_max, theta_max_type, method):
        
        if self.__cluster_data is None:
            self.get_cluster_data()
            
        if self.__true_galaxy_data is None:
            self.get_true_data()
           
        if self.__member_data is None:
            self.get_member_data()
        
        match_num_1w, match_num_2w, ind_bij = AssociationMethods.volume_match(self.__true_halo_data, 
                                                                              self.__cluster_data, 
                                                                              delta_zmax,
                                                                              theta_max, 
                                                                              theta_max_type, 
                                                                              method, 
                                                                              self.truth_catalog.cosmology,
                                                                              self.__true_galaxy_data, 
                                                                              self.__member_data)
        
        #print statistics
        print ("Number of bijective associations", AssociationStatistics.number_of_associations(ind_bij))
        print ("Number and fraction of fragmentation", AssociationStatistics.fragmentation(match_num_1w, ind_bij, method="bij"))
        print ("Number and fraction of overmerging", AssociationStatistics.overmerging(match_num_2w, ind_bij, method="bij"))
        print ("Completeness", AssociationStatistics.completeness(self.__true_halo_data, ind_bij, self.redmapper_catalog, self.truth_catalog))
        print ("Purity", AssociationStatistics.purity(self.__cluster_data, ind_bij, self.redmapper_catalog, self.truth_catalog))
        
        PlottingHelper.plot_cluster_and_halo_position(self.__true_halo_data, self.__cluster_data, match_num_1w, match_num_2w, ind_bij)
        #PlottingHelper.plot_redshift_comparison(self.__true_halo_data, self.__cluster_data, ind_bij)
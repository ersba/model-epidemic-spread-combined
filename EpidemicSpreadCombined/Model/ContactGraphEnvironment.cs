using System.Collections.Generic;
using System.IO;
using Mars.Interfaces;
using Mars.Interfaces.Environments;

namespace EpidemicSpreadCombined.Model
{
    /// <summary>
    /// Custom Environment class managing the contact graph of the agents. 
    /// </summary>
    public class ContactGraphEnvironment : IEnvironment, IModelObject
    {
        private Dictionary<int, Host> _hosts = new Dictionary<int, Host>();
        
        private Dictionary<int, List<Host>> _edges = new Dictionary<int, List<Host>>();
        
        /// <summary>
        /// Inserts a new host into the contact graph environment.
        /// The host is added to the dictionary of hosts.
        /// </summary>
        /// <param name="host"></param>
        public void Insert(Host host)
        {
            _hosts.Add(host.Index, host);
            _edges[host.Index] = new List<Host>();
        }
        
        /// <summary>
        /// Reads a CSV file containing contact edges between hosts.
        /// Each line in the CSV file represents a contact edge between two hosts, identified by their indices.
        /// If both host indices are within the agent count limit, the hosts are retrieved from the hosts dictionary and
        /// added to each other's neighbors list in the edges dictionary. This ensures that no neighbors are added that
        /// are not in the model.
        /// </summary>
        public void ReadCSV()
        {
            foreach (var line in File.ReadAllLines(Params.ContactEdgesPath))
            {
                var splitLine = line.Split(',');
                int hostIndex0 = int.Parse(splitLine[0]);
                int hostIndex1 = int.Parse(splitLine[1]);
                if (hostIndex0 < Params.AgentCount && hostIndex1 < Params.AgentCount)
                {
                    Host host0 = _hosts[hostIndex0];
                    Host host1 = _hosts[hostIndex1];
                    _edges[hostIndex0].Add(host1);
                    _edges[hostIndex1].Add(host0);
                }
            }
        }
        
        /// <summary>
        /// Retrieves the list of neighbors for a given host in the contact graph environment.
        /// If the host has neighbors, it returns the list of neighbors.
        /// If the host does not have any neighbors it returns an empty list.
        /// </summary>
        /// <param name="hostIndex">The index of the host whose neighbors are to be retrieved.</param>
        /// <returns>A list of neighbors for the given host.</returns>
        public List<Host> GetNeighbors(int hostIndex)
        {
            if (_edges.TryGetValue(hostIndex, out List<Host> neighbors))
            {
                return neighbors;
            }
            return new List<Host>();
        }
    }
}
using System.Collections.Generic;
using System.IO;
using Mars.Interfaces;
using Mars.Interfaces.Environments;

namespace EpidemicSpreadCombined.Model
{
    public class ContactGraphEnvironment : IEnvironment, IModelObject
    {
        private Dictionary<int, Host> _hosts = new Dictionary<int, Host>();
        
        private Dictionary<int, List<Host>> _edges = new Dictionary<int, List<Host>>();
        
        public void Insert(Host host)
        {
            _hosts.Add(host.Index, host);
            _edges[host.Index] = new List<Host>();
        }
        
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
using System;
using System.Linq;
using System.Reflection;

static void Dump(string title, Assembly assembly)
{
    Console.WriteLine($"Assembly: {title} => {assembly.FullName}");
    foreach (var type in assembly.GetTypes().Where(t => (t.FullName ?? string.Empty).Contains("Wex", StringComparison.OrdinalIgnoreCase))
             .OrderBy(t => t.FullName))
    {
        Console.WriteLine($"Type: {type.FullName}");
        foreach (var member in type.GetMembers(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.Static)
                     .OrderBy(m => m.Name))
        {
            Console.WriteLine($"  {member.MemberType}: {member.Name}");
        }
    }
}

var assemblies = new[]
{
    typeof(Xbim.ModelGeometry.Scene.Xbim3DModelContext).Assembly,
    Assembly.Load("Xbim.Geometry.Engine.Interop"),
    Assembly.Load("Xbim.Common"),
    Assembly.Load("Xbim.IO.Esent"),
    Assembly.Load("Xbim.IO.MemoryModel"),
    Assembly.Load("Xbim.Ifc"),
};

foreach (var asm in assemblies)
{
    Dump(asm.GetName().Name, asm);
}

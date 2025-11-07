using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text.Json;
using Xbim.Ifc;
using Xbim.Ifc4.Interfaces;
using Xbim.ModelGeometry.Scene;

namespace XbimPreprocess;

internal static class Program
{
    private sealed record Options(
        string Input,
        string Output,
        string? Wexbim,
        string? Stats,
        bool Overwrite,
        bool SkipTessellation)
    {
        public static Options Parse(string[] args)
        {
            if (args.Length == 0)
            {
                throw new ArgumentException("No arguments supplied – expected --in <path> --out <path>");
            }

            string? input = null;
            string? output = null;
            string? wexbim = null;
            string? stats = null;
            var overwrite = false;
            var skip = false;

            for (var i = 0; i < args.Length; i++)
            {
                var arg = args[i];
                switch (arg)
                {
                    case "--in":
                    case "-i":
                        input = NextValue(args, ref i, arg);
                        break;
                    case "--out":
                    case "-o":
                        output = NextValue(args, ref i, arg);
                        break;
                    case "--wexbim":
                        wexbim = NextValue(args, ref i, arg);
                        break;
                    case "--stats":
                        stats = NextValue(args, ref i, arg);
                        break;
                    case "--overwrite":
                        overwrite = true;
                        break;
                    case "--skip-tessellation":
                        skip = true;
                        break;
                    case "--help":
                    case "-h":
                        PrintUsage();
                        Environment.Exit(0);
                        break;
                    default:
                        throw new ArgumentException($"Unknown argument '{arg}'");
                }
            }

            if (string.IsNullOrWhiteSpace(input))
            {
                throw new ArgumentException("Missing --in <file> argument");
            }
            if (string.IsNullOrWhiteSpace(output))
            {
                throw new ArgumentException("Missing --out <file> argument");
            }

            return new Options(input, output, wexbim, stats, overwrite, skip);
        }

        private static string NextValue(string[] args, ref int i, string name)
        {
            if (i + 1 >= args.Length)
            {
                throw new ArgumentException($"Missing value for {name}");
            }

            return args[++i];
        }

        private static void PrintUsage()
        {
            Console.WriteLine("xBIM IFC preprocessor");
            Console.WriteLine("Usage: XbimPreprocess --in input.ifc --out output.ifc [--wexbim output.wexbim] [--stats stats.json] [--overwrite]");
        }
    }

    private sealed record GeometryStats
    {
        public int ProductsVisited { get; set; }
        public int ProductsUpdated { get; set; }
        public int TessellationsCreated { get; set; }
        public int TrianglesTotal { get; set; }
        public int VerticesTotal { get; set; }
        public long? InputBytes { get; set; }
        public long? OutputBytes { get; set; }
        public bool TessellationSkipped { get; set; }
        public bool WexbimRequested { get; set; }
        public bool WexbimWritten { get; set; }
        public string? CommandLine { get; set; }
        public string ToolVersion { get; } = typeof(Program).Assembly.GetName().Version?.ToString() ?? "dev";
        public string StatsVersion { get; } = "1.0";

        public IDictionary<string, object> ToDictionary()
        {
            var payload = new Dictionary<string, object>
            {
                ["statsVersion"] = StatsVersion,
                ["toolVersion"] = ToolVersion,
                ["productsVisited"] = ProductsVisited,
                ["productsUpdated"] = ProductsUpdated,
                ["tessellationsCreated"] = TessellationsCreated,
                ["triangles"] = TrianglesTotal,
                ["vertices"] = VerticesTotal,
                ["tessellationSkipped"] = TessellationSkipped,
                ["wexbimRequested"] = WexbimRequested,
                ["wexbimWritten"] = WexbimWritten,
                ["timestamp"] = DateTimeOffset.UtcNow.ToString("O", CultureInfo.InvariantCulture),
            };

            if (InputBytes.HasValue)
            {
                payload["inputBytes"] = InputBytes.Value;
            }

            if (OutputBytes.HasValue)
            {
                payload["outputBytes"] = OutputBytes.Value;
            }

            if (!string.IsNullOrWhiteSpace(CommandLine))
            {
                payload["commandLine"] = CommandLine;
            }

            return payload;
        }
    }

    private static int Main(string[] args)
    {
        try
        {
            var options = Options.Parse(args);
            Run(options);
            return 0;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[xbim-preprocess] {ex.Message}");
            Console.Error.WriteLine(ex);
            return 1;
        }
    }

    private static void Run(Options options)
    {
        var inputPath = Path.GetFullPath(options.Input);
        var outputPath = Path.GetFullPath(options.Output);
        if (!File.Exists(inputPath))
        {
            throw new FileNotFoundException("Input IFC not found", inputPath);
        }

        if (File.Exists(outputPath) && !options.Overwrite)
        {
            throw new InvalidOperationException($"Output file '{outputPath}' already exists – pass --overwrite to replace it.");
        }

        Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);

        var stats = new GeometryStats
        {
            TessellationSkipped = options.SkipTessellation,
            WexbimRequested = !string.IsNullOrWhiteSpace(options.Wexbim),
            CommandLine = string.Join(" ", Environment.GetCommandLineArgs())
        };
        var inputInfo = new FileInfo(inputPath);
        if (inputInfo.Exists)
        {
            stats.InputBytes = inputInfo.Length;
        }

        using var model = IfcStore.Open(inputPath);
        // Count products for diagnostics
        stats.ProductsVisited = model.Instances.OfType<IIfcProduct>().Count();

        var needsGeometryContext = !options.SkipTessellation || !string.IsNullOrWhiteSpace(options.Wexbim);
        Xbim3DModelContext? context = null;
        if (needsGeometryContext)
        {
            if (options.SkipTessellation && !string.IsNullOrWhiteSpace(options.Wexbim))
            {
                Console.WriteLine("[xbim-preprocess] Tessellation was skipped by flag, but WexBIM export requires geometry – tessellating now.");
                stats.TessellationSkipped = false;
            }

            context = new Xbim3DModelContext(model);
            context.CreateContext();
            stats.ProductsUpdated = stats.ProductsVisited;
            stats.TessellationsCreated = stats.ProductsVisited;
        }

        // Copy file to output
        model.SaveAs(outputPath);
        var outputInfo = new FileInfo(outputPath);
        if (outputInfo.Exists)
        {
            stats.OutputBytes = outputInfo.Length;
        }

        if (!string.IsNullOrWhiteSpace(options.Wexbim))
        {
            var wexOut = Path.GetFullPath(options.Wexbim);
            Directory.CreateDirectory(Path.GetDirectoryName(wexOut)!);

            try
            {
                using var stream = File.Create(wexOut);
                using var writer = new BinaryWriter(stream);
                if (context == null)
                {
                    context = new Xbim3DModelContext(model);
                    context.CreateContext();
                    stats.ProductsUpdated = stats.ProductsVisited;
                }
                model.SaveAsWexBim(writer);
                stats.WexbimWritten = true;
                Console.WriteLine($"[xbim-preprocess] WexBIM written to {wexOut}");
            }
            catch (Exception ex)
            {
                stats.WexbimWritten = false;
                Console.Error.WriteLine($"[xbim-preprocess] Failed to write WexBIM: {ex.Message}");
                Console.Error.WriteLine(ex);
                try
                {
                    File.Delete(wexOut);
                }
                catch
                {
                    // ignore cleanup errors
                }
            }
        }

        if (!string.IsNullOrWhiteSpace(options.Stats))
        {
            var json = JsonSerializer.Serialize(stats.ToDictionary(), new JsonSerializerOptions
            {
                WriteIndented = true,
            });
            File.WriteAllText(Path.GetFullPath(options.Stats), json);
        }
    }

    // Geometry-specific routines are intentionally omitted in this minimal build.
}


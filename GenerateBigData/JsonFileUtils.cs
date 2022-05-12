using System.Text.Json;
using System.Text.Json.Serialization;

namespace GenerateBigData.Utils;

public static class JsonFileUtils
{
    private static readonly JsonSerializerOptions _options =
        new() { DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull };


    public static void Write(object obj, string fileName)
    {
        var options = new JsonSerializerOptions(_options)
        {
            WriteIndented = true
        };

        if (File.Exists(fileName))
            File.Delete(fileName);

        var jsonString = JsonSerializer.Serialize(obj, options);
        File.WriteAllText(fileName, jsonString);
    }
}

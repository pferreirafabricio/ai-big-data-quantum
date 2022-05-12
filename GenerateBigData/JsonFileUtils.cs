namespace GenerateBigData.Utils;
using System;
using System.Object;

public static class JsonFileUtils
{
    private static readonly JsonSerializerSettings _options
        = new() { NullValueHandling = NullValueHandling.Ignore };

    public static void StreamWrite(object obj, string fileName)
    {
        var options = new JsonSerializerOptions(_options) 
        { 
            WriteIndented = true
        };

        using var fileStream = File.Create(fileName);
        using var utf8JsonWriter = new Utf8JsonWriter(fileStream);

        JsonSerializer.Serialize(utf8JsonWriter, obj, _options);
    }
}

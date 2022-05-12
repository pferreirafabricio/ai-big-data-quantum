using Bogus;
using GenerateBigData.DTO;
using GenerateBigData.Utils;

int numberOfStudents = int.Parse(args.FirstOrDefault() ?? "1");
string fileName = args[1] ?? "Students";

Console.WriteLine("Initializing students creation...");

var studentFakerGenerator = new Faker<Student>()
    .StrictMode(true)
    .RuleFor(o => o.Name, f => f.Name.FullName())
    .RuleFor(o => o.Avatar, f => f.Internet.Avatar())
    .RuleFor(o => o.Age, f => f.Random.Number(18, 60));

var students = new List<Student>();

for (int index = 0; index < numberOfStudents; index++)
    students.Add(studentFakerGenerator.Generate());

JsonFileUtils.Write(students, $"Samples/{fileName}.json");

Console.WriteLine("Students creation done!");

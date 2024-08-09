import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("exp1-tinyllama-model")
model = AutoModelForCausalLM.from_pretrained("exp1-tinyllama-model", torch_dtype=torch.bfloat16)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = model.to(device)

prompt = """Classify what skills are required from the following job description:

{}

Skills (separated by commas): {}
"""

inputs = tokenizer(
[
    prompt.format(
        "Backend Developer (C# ASP.Net) Pegawai (non-manajemen & non-supervisor) Sertifikat Professional, D3 (Diploma), D4 (Diploma), Sarjana (S1), Diploma Pascasarjana, Gelar Professional, Magister (S2) Penuh Waktu Asuransi Gigi;Asuransi kesehatan;Parkir;Waktu regular, Senin - Jumat;Kasual (contoh: Kaos) PersyaratanBerpengalaman minimal 3 tahun dalam mendesign atau mengembangkan aplikasi menggunakan .NET, C#, ASP.NET, Windows Forms dan SQL ServerBerpengalaman dalam menggunakan PHP dan frameworknyaBerpengalaman menggunakan GIT sebagai Version Control SystemMemahami architecture styles / API (REST, RPC)Memahami object-oriented developmentBerpengalaman dengan Agile methodologiesDeskripsi PekerjaanMembuat dan mengembangkan aplikasi menggunakan C# ASP.NET dan PHPMelakukan pengujian dan memeriksa kualitas aplikasiMerancang code untuk berbagai aplikasi .NET dan memperbaiki semua bug atau defects dalam systemMengembangkan system yang ada dan mengidentifikasi area untuk modifikasi dan improvementMengelola system serta memperbaiki semua masalah dan mempersiapkan update untuk system ", # instruction
        ""
    )
], return_tensors = "pt").to(device)

outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
print(tokenizer.batch_decode(outputs))

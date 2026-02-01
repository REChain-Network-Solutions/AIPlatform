<?php

use App\Models\OpenaiGenerator;
use Illuminate\Database\Migrations\Migration;
use Illuminate\Support\Arr;
use Illuminate\Support\Facades\Schema;
use Illuminate\Support\Facades\Storage;

return new class extends Migration
{
    public function up(): void
    {
        if (! Schema::hasTable('openai')) {
            return;
        }

        foreach ([
            'academic',
            'business',
            'customer service',
            'entertainment',
            'website',
            'advertising',
            'languages',
            'email',
            'fitness & health',
            'writer',
            'misc',
        ] as $name) {
            \App\Models\OpenaiGeneratorFilter::query()->firstOrCreate([
                'name' => $name,
            ]);
        }

        $openai = Storage::disk('extension')->get('AIWriterTemplates/database/data/openai.json');

        $data = json_decode($openai, true);

        foreach ($data as $item) {
            OpenaiGenerator::query()->firstOrCreate([
                'slug' => $item['slug'],
            ], Arr::except($item, ['slug', 'id']));
        }
    }

    public function down(): void {}
};

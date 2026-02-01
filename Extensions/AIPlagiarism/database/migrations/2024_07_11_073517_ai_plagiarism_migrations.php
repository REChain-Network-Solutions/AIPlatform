<?php

use App\Models\OpenAIGenerator;
use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    public function up(): void
    {
        if (! Schema::hasColumn('settings_two', 'plagiarism_key')) {
            Schema::table('settings_two', function (Blueprint $table) {
                $table->string('plagiarism_key')->nullable();
            });
        }

        if (Schema::hasTable('openai')) {
            OpenAIGenerator::query()->firstOrCreate([
                'slug' => 'ai_plagiarism',
            ], [
                'title'           => 'AI Plagiarism Checker',
                'description'     => 'Analyze text, comparing it against a vast database online content to identify potential plagiarism.',
                'active'          => 1,
                'questions'       => '[{\"name\":\"your_description\",\"type\":\"textarea\",\"question\":\"Description\",\"select\":\"\"}]',
                'image'           => '<svg xmlns=\"http://www.w3.org/2000/svg\" height=\"48\" viewBox=\"0 96 960 960\" width=\"48\"><path d=\"M430 896V356H200V256h560v100H530v540H430Z\"/></svg>',
                'premium'         => 0,
                'type'            => 'text',
                'prompt'          => null,
                'custom_template' => 0,
                'tone_of_voice'   => 0,
                'color'           => '#A3D6C2',
                'filters'         => 'blog',
            ]);

            OpenAIGenerator::query()->firstOrCreate([
                'slug' => 'ai_content_detect',
            ], [
                'title'           => 'AI Content Detector',
                'description'     => 'Analylze text, comparing it against a vast database online content to AI writing content.',
                'active'          => 1,
                'questions'       => '[{\"name\":\"your_description\",\"type\":\"textarea\",\"question\":\"Description\",\"select\":\"\"}]',
                'image'           => '<svg xmlns=\"http://www.w3.org/2000/svg\" height=\"48\" viewBox=\"0 96 960 960\" width=\"48\"><path d=\"M430 896V356H200V256h560v100H530v540H430Z\"/></svg>',
                'premium'         => 0,
                'type'            => 'text',
                'prompt'          => null,
                'custom_template' => 0,
                'tone_of_voice'   => 0,
                'color'           => '#A3D6C2',
                'filters'         => 'blog',
            ]);
        }
    }

    public function down(): void {}
};

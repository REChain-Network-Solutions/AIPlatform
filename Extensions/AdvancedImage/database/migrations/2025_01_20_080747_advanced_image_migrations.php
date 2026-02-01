<?php

use App\Models\OpenAIGenerator;
use Illuminate\Database\Migrations\Migration;

return new class extends Migration
{
    public function up(): void
    {
        $data = __DIR__ . './../resources/db.json';

        try {
            $json = file_get_contents($data);

            $data = json_decode($json, true);

            if (is_array($data)) {

                $slug = data_get($data, 'slug');

                $openai = OpenAIGenerator::whereSlug('$slug')->exists();

                if ($openai) {
                    return;
                }

                OpenAIGenerator::query()->create($data);
            }
        } catch (\Exception $exception) {
        }
    }

    public function down(): void {}
};

<?php

use App\Domains\Entity\Models\Entity;
use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\DB;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    /**
     * Run the migrations.
     */
    public function up(): void
    {
        Schema::table('entities', function (Blueprint $table) {
            $table->string('image')->nullable()->after('title');
        });

        $this->seedImages();
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::table('entities', function (Blueprint $table) {
            $table->dropColumn('image');
        });
    }

    public function seedImages(): void
    {
        // Map engine names to their corresponding logos
        $engineLogos = [
            'openai'           => 'upload/enginelogo/openai.svg',
            'deep_seek'        => 'upload/enginelogo/deepseek_logo.svg',
            'anthropic'        => 'upload/enginelogo/claude_logo.svg',
            'gemini'           => 'upload/enginelogo/gemini_logo.svg',
            'perplexity'       => 'upload/enginelogo/Perplexity_logo.svg',
            'x_ai'             => 'upload/enginelogo/Grok_logo.svg',
            'stable_diffusion' => 'upload/enginelogo/stability.svg',
            'amazon'           => 'upload/enginelogo/amazon_logo.svg',
            'mistral'          => 'upload/enginelogo/Mistral_logo.svg',
            'qwen'             => 'upload/enginelogo/Qwen_logo.svg',
            'nvidia'           => 'upload/enginelogo/nvidia_logo_new_2.svg',
            'open_router'      => 'upload/enginelogo/openrouter.svg',
        ];

        // Get unique engines and their entity counts from entities table
        $enginesWithCounts = Entity::select('engine', DB::raw('count(*) as entity_count'))
            ->groupBy('engine')
            ->get();

        foreach ($enginesWithCounts as $engineData) {
            $engine = $engineData->engine;

            // Convert EngineEnum to string if needed
            $engineValue = is_object($engine) ? $engine->value : $engine;

            if (isset($engineLogos[$engineValue])) {
                // Update all entities for this engine in entities table
                $updated = Entity::where('engine', $engine)
                    ->whereNull('image') // Only update if image is null
                    ->update(['image' => $engineLogos[$engineValue]]);

            }
        }
    }
};

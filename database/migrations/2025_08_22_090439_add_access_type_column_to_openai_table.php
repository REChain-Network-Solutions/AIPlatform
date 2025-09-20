<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    /**
     * Run the migrations.
     */
    public function up(): void
    {
        Schema::table('openai', function (Blueprint $table) {
            $table->string('access_type', 20)->default(\App\Enums\AccessType::REGULAR->value)->index();
        });

        try {
            \App\Models\OpenAIGenerator::query()->each(function ($generator) {
                $generator->access_type = $generator->premium ? \App\Enums\AccessType::PREMIUM->value : \App\Enums\AccessType::REGULAR->value;
                $generator->save();
            });

            \App\Models\OpenaiGeneratorChatCategory::query()->each(function ($generator) {
                if (! $generator->plan) {
                    $generator->plan = \App\Enums\AccessType::REGULAR->value;
                    $generator->save();
                }
            });
        } catch (\Throwable $th) {
        }
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::table('openai', function (Blueprint $table) {
            $table->dropColumn('access_type');
        });
    }
};

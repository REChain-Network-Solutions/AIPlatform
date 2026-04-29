<?php

use App\Enums\AccessType;
use App\Models\OpenAIGenerator;
use App\Models\OpenaiGeneratorChatCategory;
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
            $table->string('access_type', 20)->default(AccessType::REGULAR->value)->index();
        });

        try {
            OpenAIGenerator::query()->each(function ($generator) {
                $generator->access_type = $generator->premium ? AccessType::PREMIUM->value : AccessType::REGULAR->value;
                $generator->save();
            });

            OpenaiGeneratorChatCategory::query()->each(function ($generator) {
                if (! $generator->plan) {
                    $generator->plan = AccessType::REGULAR->value;
                    $generator->save();
                }
            });
        } catch (Throwable $th) {
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

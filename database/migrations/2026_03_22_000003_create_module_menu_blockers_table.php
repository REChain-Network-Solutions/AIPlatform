<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    public function up(): void
    {
        if (Schema::hasTable('module_menu_blockers')) {
            return;
        }

        Schema::create('module_menu_blockers', function (Blueprint $table) {
            $table->id();
            $table->string('module_name')->index();
            $table->string('blocker_key')->index();
            $table->string('severity', 32)->default('warn');
            $table->string('target_profile', 64)->nullable();
            $table->string('source_file')->nullable();
            $table->text('message');
            $table->boolean('is_auto_fixable')->default(false);
            $table->json('details_json')->nullable();
            $table->timestamps();
        });
    }

    public function down(): void
    {
        Schema::dropIfExists('module_menu_blockers');
    }
};

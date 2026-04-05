<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration {
    public function up(): void
    {
        if (Schema::hasTable('module_visibility_audits')) {
            return;
        }

        Schema::create('module_visibility_audits', function (Blueprint $table) {
            $table->id();
            $table->string('module_name')->nullable();
            $table->string('source_file')->nullable();
            $table->string('outcome')->default('analysis');
            $table->json('payload')->nullable();
            $table->timestamps();
        });
    }

    public function down(): void
    {
        Schema::dropIfExists('module_visibility_audits');
    }
};

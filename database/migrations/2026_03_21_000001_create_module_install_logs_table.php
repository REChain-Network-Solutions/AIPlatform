<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration {
    public function up(): void
    {
        Schema::create('module_install_logs', function (Blueprint $table) {
            $table->id();
            $table->string('event', 50);
            $table->string('module_name')->nullable();
            $table->string('uploaded_filename')->nullable();
            $table->string('detected_type', 50)->nullable();
            $table->string('installability_status', 50)->nullable();
            $table->string('recommended_mode', 50)->nullable();
            $table->json('blocking_issues')->nullable();
            $table->json('warnings')->nullable();
            $table->json('checks')->nullable();
            $table->json('package_summary')->nullable();
            $table->text('suggested_action')->nullable();
            $table->unsignedBigInteger('created_by')->nullable();
            $table->timestamps();

            $table->index(['module_name', 'event']);
            $table->index(['detected_type', 'installability_status']);
        });
    }

    public function down(): void
    {
        Schema::dropIfExists('module_install_logs');
    }
};

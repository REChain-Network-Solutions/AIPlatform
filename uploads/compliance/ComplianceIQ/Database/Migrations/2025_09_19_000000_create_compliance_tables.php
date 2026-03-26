<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration {
    public function up(): void
    {
        Schema::create('compliance_reports', function (Blueprint $table) {
            $table->id();
            $table->string('title');
            $table->date('period_start');
            $table->date('period_end');
            $table->enum('status', ['draft','in_review','signed_off'])->default('draft');
            $table->foreignId('signed_off_by')->nullable()->constrained('users')->nullOnDelete();
            $table->timestamp('signed_off_at')->nullable();
            $table->json('filters')->nullable();
            $table->json('summary')->nullable();
            $table->timestamps();
        });

        Schema::create('compliance_annotations', function (Blueprint $table) {
            $table->id();
            $table->foreignId('report_id')->constrained('compliance_reports')->cascadeOnDelete();
            $table->foreignId('user_id')->constrained('users')->cascadeOnDelete();
            $table->text('note');
            $table->timestamps();
        });

        Schema::create('compliance_hashes', function (Blueprint $table) {
            $table->id();
            $table->string('hashable_type');
            $table->unsignedBigInteger('hashable_id');
            $table->string('sha256', 64);
            $table->timestamp('computed_at');
            $table->enum('status', ['valid','mismatch','unknown'])->default('unknown');
            $table->index(['hashable_type','hashable_id']);
        });
    }

    public function down(): void
    {
        Schema::dropIfExists('compliance_hashes');
        Schema::dropIfExists('compliance_annotations');
        Schema::dropIfExists('compliance_reports');
    }
};

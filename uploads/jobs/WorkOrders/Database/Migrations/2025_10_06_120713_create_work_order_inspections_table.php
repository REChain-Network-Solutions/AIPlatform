<?php
use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;
return new class extends Migration {
    public function up(): void {
        if (!Schema::hasTable('work_order_inspections')) {
            Schema::create('work_order_inspections', function (Blueprint $table) {
                $table->id();
                $table->unsignedBigInteger('work_order_id');
                $table->unsignedBigInteger('inspection_id')->nullable();
                $table->string('template_name')->nullable();
                $table->unsignedBigInteger('completed_by')->nullable();
                $table->timestamp('completed_at')->nullable();
                $table->string('pdf_path')->nullable();
                $table->timestamps();
                $table->index(['work_order_id']);
            });
        }
    }
    public function down(): void { Schema::dropIfExists('work_order_inspections'); }
};
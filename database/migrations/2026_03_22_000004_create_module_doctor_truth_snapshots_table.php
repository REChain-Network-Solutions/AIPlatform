<?php
use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;
return new class extends Migration {
    public function up(): void {
        Schema::create('module_doctor_truth_snapshots', function (Blueprint $table) {
            $table->id();
            $table->string('module_name');
            $table->string('status_code')->nullable();
            $table->json('payload')->nullable();
            $table->timestamps();
        });
    }
    public function down(): void {
        Schema::dropIfExists('module_doctor_truth_snapshots');
    }
};

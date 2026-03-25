<?php
use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration {
    public function up(): void {
        Schema::create('wo_service_appointments', function (Blueprint $table) {
            $table->id();
            $table->unsignedBigInteger('work_order_id')->index();
            $table->unsignedBigInteger('technician_id')->nullable()->index();
            $table->dateTime('starts_at')->nullable();
            $table->dateTime('ends_at')->nullable();
            $table->string('location')->nullable();
            $table->string('status')->default('scheduled'); // scheduled, in_progress, done, canceled
            $table->timestamps();
        });
    }
    public function down(): void {
        Schema::dropIfExists('wo_service_appointments');
    }
};

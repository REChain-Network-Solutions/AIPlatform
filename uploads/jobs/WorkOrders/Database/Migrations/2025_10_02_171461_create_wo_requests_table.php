<?php
use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration {
    public function up(): void {
        Schema::create('wo_requests', function (Blueprint $table) {
            $table->id();
            $table->unsignedBigInteger('work_order_id')->index();
            $table->unsignedBigInteger('requested_by_id')->nullable()->index();
            $table->string('channel')->nullable(); // phone, web, email, app
            $table->text('description')->nullable();
            $table->timestamps();
        });
    }
    public function down(): void {
        Schema::dropIfExists('wo_requests');
    }
};

<?php
use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;
return new class extends Migration {
    public function up(): void {
        if (!Schema::hasTable('work_order_permits')) {
            Schema::create('work_order_permits', function (Blueprint $table) {
                $table->id();
                $table->unsignedBigInteger('work_order_id');
                $table->string('type')->nullable();
                $table->string('status')->default('pending');
                $table->string('permit_number')->nullable();
                $table->date('valid_from')->nullable();
                $table->date('valid_to')->nullable();
                $table->timestamps();
                $table->index(['work_order_id']);
            });
        }
    }
    public function down(): void { Schema::dropIfExists('work_order_permits'); }
};
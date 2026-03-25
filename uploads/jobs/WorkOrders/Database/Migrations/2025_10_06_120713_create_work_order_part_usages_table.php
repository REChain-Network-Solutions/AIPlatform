<?php
use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;
return new class extends Migration {
    public function up(): void {
        if (!Schema::hasTable('work_order_part_usages')) {
            Schema::create('work_order_part_usages', function (Blueprint $table) {
                $table->id();
                $table->unsignedBigInteger('work_order_id');
                $table->unsignedBigInteger('item_id')->nullable();
                $table->string('item_name')->nullable();
                $table->decimal('qty', 12, 3)->default(0);
                $table->decimal('unit_price', 12, 2)->nullable();
                $table->string('source_location')->nullable();
                $table->timestamps();
                $table->index(['work_order_id']);
            });
        }
    }
    public function down(): void { Schema::dropIfExists('work_order_part_usages'); }
};
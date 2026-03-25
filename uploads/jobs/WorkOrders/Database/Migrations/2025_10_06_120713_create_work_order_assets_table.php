<?php
use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;
return new class extends Migration {
    public function up(): void {
        if (!Schema::hasTable('work_order_assets')) {
            Schema::create('work_order_assets', function (Blueprint $table) {
                $table->id();
                $table->unsignedBigInteger('work_order_id');
                $table->unsignedBigInteger('asset_id');
                $table->timestamps();
                $table->unique(['work_order_id','asset_id']);
            });
        }
    }
    public function down(): void { Schema::dropIfExists('work_order_assets'); }
};
<?php
use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;
return new class extends Migration {
    public function up(): void {
        if (!Schema::hasTable('work_order_recurrences')) {
            Schema::create('work_order_recurrences', function (Blueprint $table) {
                $table->id();
                $table->unsignedBigInteger('work_order_id');
                $table->string('rrule'); // iCal RRULE
                $table->timestamp('next_run_at')->nullable();
                $table->timestamp('last_run_at')->nullable();
                $table->boolean('is_active')->default(true);
                $table->timestamps();
                $table->index(['work_order_id']);
            });
        }
    }
    public function down(): void { Schema::dropIfExists('work_order_recurrences'); }
};
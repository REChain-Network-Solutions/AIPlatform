<?php
use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration {
    public function up(): void {
        Schema::create('workorders_settings', function (Blueprint $table) {
            $table->id();
            $table->boolean('auto_convert_on_complete')->default(false);
            $table->timestamps();
        });
    }
    public function down(): void {
        Schema::dropIfExists('workorders_settings');
    }
};

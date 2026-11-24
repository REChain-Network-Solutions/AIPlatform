<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    public function up(): void
    {
        Schema::create('shared_credit_costs', function (Blueprint $table) {
            $table->id();
            $table->string('entity_key')->unique();
            $table->string('engine_key')->nullable();
            $table->string('feature_type')->nullable();
            $table->decimal('base_cost', 10, 4);
            $table->decimal('quality_high_multiplier', 6, 2)->default(1.00);
            $table->decimal('quality_low_multiplier', 6, 2)->default(1.00);
            $table->boolean('is_active')->default(true);
            $table->timestamps();
        });
    }

    public function down(): void
    {
        Schema::dropIfExists('shared_credit_costs');
    }
};

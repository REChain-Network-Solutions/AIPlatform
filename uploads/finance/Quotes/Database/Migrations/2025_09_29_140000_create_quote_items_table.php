<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    public function up(): void
    {
        if (!Schema::hasTable('quote_items')) {
            Schema::create('quote_items', function (Blueprint $table) {
                $table->bigIncrements('id');
                $table->unsignedBigInteger('quote_id');
                $table->unsignedBigInteger('item_id')->nullable();
                $table->string('description');
                $table->decimal('qty', 10, 2)->default(1);
                $table->decimal('unit_price', 12, 2)->default(0);
                $table->decimal('tax_rate', 8, 4)->default(0); // e.g., 0.1000 = 10%
                $table->decimal('line_total', 12, 2)->default(0);
                $table->timestamps();
                $table->index('quote_id');
            });
        }
    }

    public function down(): void
    {
        Schema::dropIfExists('quote_items');
    }
};

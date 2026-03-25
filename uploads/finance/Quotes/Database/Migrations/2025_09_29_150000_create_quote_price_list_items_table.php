<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    public function up(): void
    {
        if (!Schema::hasTable('quote_price_list_items')) {
            Schema::create('quote_price_list_items', function (Blueprint $table) {
                $table->bigIncrements('id');
                $table->unsignedBigInteger('price_list_id');
                $table->string('item_name');
                $table->decimal('unit_price', 12, 2)->default(0);
                $table->decimal('tax_rate', 8, 4)->default(0);
                $table->timestamps();
                $table->index('price_list_id');
            });
        }
    }

    public function down(): void
    {
        Schema::dropIfExists('quote_price_list_items');
    }
};

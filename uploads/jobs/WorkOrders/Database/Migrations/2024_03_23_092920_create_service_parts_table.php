<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    /**
     * Run the migrations.
     *
     * @return void
     */
    public function up()
    {
        Schema::create('service_parts', function (Blueprint $table) {
            $table->id();
            $table->string('title')->nullable();
            $table->string('sku')->nullable();
            $table->string('unit')->nullable();
            $table->integer('price')->default(0);
            $table->text('description')->nullable();
            $table->string('type')->nullable();
            $table->integer('parent_id')->default(0);
            $table->timestamps();
        });
    }

    /**
     * Reverse the migrations.
     *
     * @return void
     */
    public function down()
    {
        Schema::dropIfExists('service_parts');
    }
};

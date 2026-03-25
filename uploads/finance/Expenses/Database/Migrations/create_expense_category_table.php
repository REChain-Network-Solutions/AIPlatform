<?php


/**
 * Author: Emmanuel Paul Mnzava
 * Twitter: @epmnzava
 * Github:https://github.com/dbrax/income-expense
 * Email: epmnzava@gmail.com
 * 
 */

use Illuminate\Support\Facades\Schema;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Database\Migrations\Migration;

class CreateExpenseCategoryTable extends Migration
{
    /**
     * Run the migrations.
     *
     * @return void
     */
    public function up()
    {
        Schema::create('expense_category', function (Blueprint $table) {
            $table->bigIncrements('id');
            $table->string('category');
            $table->string('slug')->nullable();
            $table->text('description')->nullable();
            $table->date('date');




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
        Schema::dropIfExists('expense_category');
    }
}

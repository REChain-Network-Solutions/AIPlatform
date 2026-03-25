<?php

use Illuminate\Support\Facades\Schema;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Database\Migrations\Migration;

class CreateTasksTable extends Migration
{
    /**
     * Run the migrations.
     *
     * @return void
     */
    public function up()
    {
        if(!Schema::hasTable('tasks'))
        {
            Schema::create('tasks', function (Blueprint $table) {
                $table->bigIncrements('id');
                $table->string('title');
                $table->string('priority');
                $table->text('description');
                $table->dateTime('start_date')->nullable();
                $table->dateTime('due_date')->nullable();
                $table->string('assign_to');
                $table->integer('project_id');
                $table->string('milestone_id')->nullable();
                $table->string('status')->default('todo');
                $table->integer('order')->default(0);
                $table->integer('workspace');
                $table->timestamps();
            });
        }
    }

    /**
     * Reverse the migrations.
     *
     * @return void
     */
    public function down()
    {
        Schema::dropIfExists('tasks');
    }
}

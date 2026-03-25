<?php

namespace Workdo\Taskly\Database\Seeders;

use Illuminate\Database\Seeder;
use Illuminate\Database\Eloquent\Model;
use Workdo\AIAssistant\Entities\AssistantTemplate;

class AIAssistantTemplateListTableSeeder extends Seeder
{
    /**
     * Run the database seeds.
     *
     * @return void
     */
    public function run()
    {
        Model::unguard();

        $defaultTemplate = [
            [
                'template_name'=>'name',
                'template_module'=>'project',
                'prompt'=> "Create creative product names:  ##description## \n\nSeed words: ##keywords## \n\n" ,
                'field_json'=>'{"field":[{"label":"Seed words","placeholder":"e.g.  fast, healthy, compact","field_type":"text_box","field_name":"keywords"},{"label":"Product Description","placeholder":"e.g. Provide product details","field_type":"textarea","field_name":"description"}]}',
                'is_tone'=> 0,
            ],
            [
                'template_name'=>'description',
                'template_module'=>'project',
                'prompt'=> "Write a short and innovative description for Project  which Topic is:\n ##title## \n\nnProject Information:\n ##description##" ,
                'field_json'=>'{"field":[{"label":"Project Name","placeholder":"Project Name","field_type":"text_box","field_name":"title"},{"label":"Project Information","placeholder":"Project Information","field_type":"textarea","field_name":"description"}]}',
                'is_tone'=> 0,
            ],
            [
                'template_name'=>'title',
                'template_module'=>'milestone',
                'prompt'=> "Generate a milestone name for a ##project_name##,specifically related to ##instruction##." ,
                'field_json'=>'{"field":[{"label":"Milestone Description","placeholder":"e.g.","field_type":"textarea","field_name":"description"},{"label":" Instruction","placeholder":"e.g.","field_type":"textarea","field_name":"instruction"}]}',
                'is_tone'=> 0,
            ],
            [
                'template_name'=>'summary',
                'template_module'=>'milestone',
                'prompt'=>"Generate a strictly one line  and valuable milestone summary for a  ##project_name##, specifically related to ##milestone_name##.",
                'field_json'=>'{"field":[{"label":"Project name","placeholder":"e.g.","field_type":"text_box","field_name":"project_name"},{"label":"Milestone Name","placeholder":"e.g.","field_type":"text_box","field_name":"milestone_name"}]}',
                'is_tone'=> 0,
            ],
            [
                'template_name'=>'title',
                'template_module'=>'project task',
                'prompt'=>"Generate a task name for a project in an ##project_name##, specifically related to ##instruction##.",
                'field_json'=>'{"field":[{"label":"Project name","placeholder":"e.g.Solving Problems","field_type":"text_box","field_name":"project_name"},{"label":"Task Instruction","placeholder":"e.g.Data Analysis","field_type":"textarea","field_name":"instruction"}]}',
                'is_tone'=> 0,
            ],
            [
                'template_name'=>'description',
                'template_module'=>'project task',
                'prompt'=>"Generate short task description for a project in an ##project_name##, specifically related to ##instruction##.",
                'field_json'=>'{"field":[{"label":"Project name","placeholder":"e.g.","field_type":"text_box","field_name":"project_name"},{"label":"Task Instruction","placeholder":"e.g.","field_type":"textarea","field_name":"instruction"}]}',
                'is_tone'=> 0,
            ],
            [
                'template_name'=>'title',
                'template_module'=>'project bug',
                'prompt'=>"You are a software developer working on a platform or service, and you're experiencing a bug where ##description##. You need to come up with a descriptive bug title for this issue. Please generate a few bug titles that could be used to report this problem.",
                'field_json'=>'{"field":[{"label":"Description of Bug","placeholder":"e.g.","field_type":"textarea","field_name":"description"}]}',
                'is_tone'=> 0,
            ],
            [
                'template_name'=>'description',
                'template_module'=>'project bug',
                'prompt'=>"generate description of bug for a ##bug_name##  in ##project_name##",
                'field_json'=>'{"field":[{"label":"Project name","placeholder":"e.g.","field_type":"text_box","field_name":"project_name"},{"label":"Bug Name","placeholder":"e.g.","field_type":"text_box","field_name":"bug_name"}]}',
                'is_tone'=> 0,
            ],
        ];
        foreach($defaultTemplate as $temp)
        {
            $check = AssistantTemplate::where('template_module',$temp['template_module'])->where('module','Taskly')->where('template_name',$temp['template_name'])->exists();
            if(!$check)
            {
                AssistantTemplate::create(
                    [
                        'template_name' => $temp['template_name'],
                        'template_module' => $temp['template_module'],
                        'module' => 'Taskly',
                        'prompt' => $temp['prompt'],
                        'field_json' => $temp['field_json'],
                        'is_tone' => $temp['is_tone'],
                        "created_at" => date('Y-m-d H:i:s'),
                        "updated_at" => date('Y-m-d H:i:s')
                    ]
                );
            }
        }
    }
}

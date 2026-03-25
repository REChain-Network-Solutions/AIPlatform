<?php

namespace Workdo\Taskly\Database\Seeders;

use Illuminate\Database\Seeder;
use Illuminate\Database\Eloquent\Model;
use Workdo\LandingPage\Entities\MarketplacePageSetting;


class MarketPlaceSeederTableSeeder extends Seeder
{
    /**
     * Run the database seeds.
     *
     * @return void
     */
    public function run()
    {
        Model::unguard();

        $data['product_main_banner'] = '';
        $data['product_main_status'] = 'on';
        $data['product_main_heading'] = 'Project';
        $data['product_main_description'] = '<p>Get an overview of the total number of projects, tasks, bugs, and members. A visual representation of the task overview and project status can help you estimate the progress on each task. Lastly, you can check the top due task.</p>';
        $data['product_main_demo_link'] = '#';
        $data['product_main_demo_button_text'] = 'View Live Demo';
        $data['dedicated_theme_heading'] = 'ONE TOOL TO MANAGE ALL YOUR PROJECTS AND PEOPLE';
        $data['dedicated_theme_description'] = '<p>Simple, yet powerful. Manage projects, work with clients, and collaborate with team members. All in one place.Create new projects and assign teams to each project. Add multiple members to share the projects with clients. You can edit permissions and controls to manage client access.</p>';
        $data['dedicated_theme_sections'] = '[{"dedicated_theme_section_image":"","dedicated_theme_section_heading":"Feasibility in finding tasks and projects with a search tab on the top","dedicated_theme_section_description":"<p>Attach cost and summary to milestones and change the status through the drop-down menu. Get a tab on recent activities of a project and also a graph about progress. Along with that, you can check your tasks&rsquo; details under project details with the help of the Gantt Chart.<\/p>","dedicated_theme_section_cards":{"1":{"title":"Collaborate With Anyone, Anywhere.","description":"Easily assign tasks, projects, and milestones to team members. Chat with your team in real-time, exchange files, and brainstorm ideas. Invite clients to discuss projects and deliver results."},"2":{"title":"Easily Manage All Your Projects and Keep Your Business Growing.","description":"Got a big team or working on multiple projects at once? Manage task priorities or even create additional workspaces and use the built-in permission system to separate core projects. Make your team more effective by helping them avoid confusion ensuring they always know what to focus on."},"3":{"title":"Kanban Task Management","description":"Whether you need a simple tool to track your tasks, are a Kanban fan, want to create Gantt Charts or are looking for a convenient tool to track your projects - Taskly got you covered."}}},{"dedicated_theme_section_image":"","dedicated_theme_section_heading":"Creating milestones and assigning subtasks","dedicated_theme_section_description":"<p>Add a new task to an already existing project and prioritize them according to the need of urgency. Assign the task to team members and set a due date for task completion. Add comments to the task and create a sub-task for ease of completion. Attach necessary files in a required task.<\/p>","dedicated_theme_section_cards":{"1":{"title":"Bugs Resolution","description":"Create new bugs and assign users and priority to them. You can write a note in the text box for the bug description. Also, the status of each bug could be changed through an easy drop-down and Kanban drag system."},"2":{"title":"Project Milestone","description":"Go beyond short-term project management. Use project milestones to help you define and track progress towards long-term goals."},"3":{"title":"Kanban Layout","description":"Get a better visual overview of all your projects, tasks, and assigned team members. Boost team productivity."}}},{"dedicated_theme_section_image":"","dedicated_theme_section_heading":"Project Task Prioritization and Commenting","dedicated_theme_section_description":"<p>Prioritize tasks to ensure that you and your team members always focus on the most important task in the pipeline. Leave relevant comments, ask questions, and get answers right on the task or project.<\/p>","dedicated_theme_section_cards":{"1":{"title":null,"description":null}}},{"dedicated_theme_section_image":"","dedicated_theme_section_heading":"Manage Your Clients With Ease","dedicated_theme_section_description":"<p>Manage your clients&rsquo; details, collaborate with them, discuss projects, and help them watch your progress. Make it easy for them to pay you for your work.<\/p>","dedicated_theme_section_cards":{"1":{"title":null,"description":null}}}]';
        $data['dedicated_theme_sections_heading'] = '';
        $data['screenshots'] = '[{"screenshots":"","screenshots_heading":"Project"},{"screenshots":"","screenshots_heading":"Project"},{"screenshots":"","screenshots_heading":"Project"},{"screenshots":"","screenshots_heading":"Project"},{"screenshots":"","screenshots_heading":"Project"}]';
        $data['addon_heading'] = 'Why choose dedicated modulesfor Your Business?';
        $data['addon_description'] = '<p>With Dash, you can conveniently manage all your business functions from a single location.</p>';
        $data['addon_section_status'] = 'on';
        $data['whychoose_heading'] = 'Why choose dedicated modulesfor Your Business?';
        $data['whychoose_description'] = '<p>With Dash, you can conveniently manage all your business functions from a single location.</p>';
        $data['pricing_plan_heading'] = 'Empower Your Workforce with DASH';
        $data['pricing_plan_description'] = '<p>Access over Premium Add-ons for Accounting, HR, Payments, Leads, Communication, Management, and more, all in one place!</p>';
        $data['pricing_plan_demo_link'] = '#';
        $data['pricing_plan_demo_button_text'] = 'View Live Demo';
        $data['pricing_plan_text'] = '{"1":{"title":"Pay-as-you-go"},"2":{"title":"Unlimited installation"},"3":{"title":"Secure cloud storage"}}';
        $data['whychoose_sections_status'] = 'on';
        $data['dedicated_theme_section_status'] = 'on';

        foreach($data as $key => $value){
            if(!MarketplacePageSetting::where('name', '=', $key)->where('module', '=', 'Taskly')->exists()){
                MarketplacePageSetting::updateOrCreate(
                [
                    'name' => $key,
                    'module' => 'Taskly'

                ],
                [
                    'value' => $value
                ]);
            }
        }
    }
}

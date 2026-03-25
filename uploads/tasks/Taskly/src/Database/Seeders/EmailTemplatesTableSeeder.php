<?php

namespace Workdo\Taskly\Database\Seeders;


use App\Models\EmailTemplate;
use App\Models\EmailTemplateLang;
use Illuminate\Database\Seeder;
use Illuminate\Database\Eloquent\Model;

class EmailTemplatesTableSeeder extends Seeder
{
    /**
     * Run the database seeds.
     *
     * @return void
     */
    public function run()
    {

        Model::unguard();
        $emailTemplate = [
            'User Invited',
            'Project Assigned',
        ];
        $defaultTemplate = [
           'User Invited' => [
                'subject' =>'New Project Invitation',
                'variables' => '{
                    "App Url": "app_url",
                    "App Name": "app_name",
                    "Company Name": "company_name",
                    "Project Creater Name": "project_creater_name",
                    "User": "name",
                    "Project": "project",
                    "URL": "url"
                    
                  }',

                    'lang' => [

                        'ar' => '<p><strong>مرحبًا,{name} </strong></p>
                        <p>أنت مدعو إلى مشروع جديد {project} بواسطة {project_creater_name} .</p>
                        <p style="text-align: left;" align="center">لا تتردد في التواصل معنا إذا كان لديك أي أسئلة.</p>
                        <p style="text-align: center;" align="center"><span style="font-size: 18pt;"><a style="background: #6676ef; color: #ffffff; font-family: "Open Sans", Helvetica, Arial, sans-serif; font-weight: normal; line-height: 120%; margin: 0px; text-decoration: none; text-transform: none;" href="{url}" target="_blank" rel="noopener"> <strong style="color: white; font-weight: bold; text: white;">افتح المشروع</strong> </a></span></p>
                        <p>شكرًا لك</p>
                        <p>{company_name}</p>
                        <p><span style="color: #000000; font-family: "Open Sans", sans-serif; font-size: 14px; background-color: #ffffff;"><strong>{app_name}</strong></span></p>',

                        'da' => '<p><strong>Hej,{name} </strong></p>
                        <p>Du inviteres ind i nyt projekt {project} ved {project_creater_name} .</p>
                        <p style="text-align: left;" align="center">Du er velkommen til at kontakte os, hvis du har spørgsmål.</p>
                        <p style="text-align: center;" align="center"><span style="font-size: 18pt;"><a style="background: #6676ef; color: #ffffff; font-family: "Open Sans", Helvetica, Arial, sans-serif; font-weight: normal; line-height: 120%; margin: 0px; text-decoration: none; text-transform: none;" href="{url}" target="_blank" rel="noopener"> <strong style="color: white; font-weight: bold; text: white;">Åbn projekt</strong> </a></span></p>
                        <p>Tak</p>
                        <p>{company_name}</p>
                        <p><span style="color: #000000; font-family: "Open Sans", sans-serif; font-size: 14px; background-color: #ffffff;"><strong>{app_name}</strong></span></p>',

                        'de' => '<p><strong>Hallo,{name} </strong></p>
                        <p>Sie werden zu einem neuen Projekt eingeladen {project} von {project_creater_name} .</p>
                        <p style="text-align: left;" align="center">Wenn Sie Fragen haben, können Sie sich jederzeit an uns wenden.</p>
                        <p style="text-align: center;" align="center"><span style="font-size: 18pt;"><a style="background: #6676ef; color: #ffffff; font-family: "Open Sans", Helvetica, Arial, sans-serif; font-weight: normal; line-height: 120%; margin: 0px; text-decoration: none; text-transform: none;" href="{url}" target="_blank" rel="noopener"> <strong style="color: white; font-weight: bold; text: white;">Projekt öffnen</strong> </a></span></p>
                        <p>Danke</p>
                        <p>{company_name}</p>
                        <p><span style="color: #000000; font-family: "Open Sans", sans-serif; font-size: 14px; background-color: #ffffff;"><strong>{app_name}</strong></span></p>',

                        'en' => '<p><strong>Hello,{name} </strong></p>
                        <p>You are invited into new project {project} by {project_creater_name} .</p>
                        <p style="text-align: left;" align="center">Feel free to reach out if you have any questions.</p>
                        <p style="text-align: center;" align="center"><span style="font-size: 18pt;"><a style="background: #6676ef; color: #ffffff; font-family: "Open Sans", Helvetica, Arial, sans-serif; font-weight: normal; line-height: 120%; margin: 0px; text-decoration: none; text-transform: none;" href="{url}" target="_blank" rel="noopener"> <strong style="color: white; font-weight: bold; text: white;">Open Project</strong> </a></span></p>
                        <p>Thank you</p>
                        <p>{company_name}</p>
                        <p><span style="color: #000000; font-family: "Open Sans", sans-serif; font-size: 14px; background-color: #ffffff;"><strong>{app_name}</strong></span></p>',

                        'es' => '<p><strong>Hola,{name} </strong></p>
                        <p>Estás invitado a un nuevo proyecto. {project} por {project_creater_name} .</p>
                        <p style="text-align: left;" align="center">No dude en comunicarse si tiene alguna pregunta.</p>
                        <p style="text-align: center;" align="center"><span style="font-size: 18pt;"><a style="background: #6676ef; color: #ffffff; font-family: "Open Sans", Helvetica, Arial, sans-serif; font-weight: normal; line-height: 120%; margin: 0px; text-decoration: none; text-transform: none;" href="{url}" target="_blank" rel="noopener"> <strong style="color: white; font-weight: bold; text: white;">Abrir proyecto</strong> </a></span></p>
                        <p>Gracias</p>
                        <p>{company_name}</p>
                        <p><span style="color: #000000; font-family: "Open Sans", sans-serif; font-size: 14px; background-color: #ffffff;"><strong>{app_name}</strong></span></p>',

                        'fr' => '<p><strong>Bonjour,{name} </strong></p>
                        <p>Vous êtes invité dans un nouveau projet {project} par {project_creater_name} .</p>
                        <p style="text-align: left;" align="center">N hésitez pas à nous contacter si vous avez des questions.</p>
                        <p style="text-align: center;" align="center"><span style="font-size: 18pt;"><a style="background: #6676ef; color: #ffffff; font-family: "Open Sans", Helvetica, Arial, sans-serif; font-weight: normal; line-height: 120%; margin: 0px; text-decoration: none; text-transform: none;" href="{url}" target="_blank" rel="noopener"> <strong style="color: white; font-weight: bold; text: white;">Projet ouvert</strong> </a></span></p>
                        <p>Merci</p>
                        <p>{company_name}</p>
                        <p><span style="color: #000000; font-family: "Open Sans", sans-serif; font-size: 14px; background-color: #ffffff;"><strong>{app_name}</strong></span></p>',

                        'it' => '<p><strong>Ciao,{name} </strong></p>
                        <p>Sei invitato a un nuovo progetto {project} di {project_creater_name} .</p>
                        <p style="text-align: left;" align="center">Sentiti libero di contattarci se hai domande.</p>
                        <p style="text-align: center;" align="center"><span style="font-size: 18pt;"><a style="background: #6676ef; color: #ffffff; font-family: "Open Sans", Helvetica, Arial, sans-serif; font-weight: normal; line-height: 120%; margin: 0px; text-decoration: none; text-transform: none;" href="{url}" target="_blank" rel="noopener"> <strong style="color: white; font-weight: bold; text: white;">Apri progetto</strong> </a></span></p>
                        <p>Grazie</p>
                        <p>{company_name}</p>
                        <p><span style="color: #000000; font-family: "Open Sans", sans-serif; font-size: 14px; background-color: #ffffff;"><strong>{app_name}</strong></span></p>',

                        'ja' => '<p><strong>こんにちは,{name} </strong></p>
                        <p>新しいプロジェクトに招待されています {project} による {project_creater_name} .</p>
                        <p style="text-align: left;" align="center">ご質問がございましたら、お気軽にお問い合わせください。</p>
                        <p style="text-align: center;" align="center"><span style="font-size: 18pt;"><a style="background: #6676ef; color: #ffffff; font-family: "Open Sans", Helvetica, Arial, sans-serif; font-weight: normal; line-height: 120%; margin: 0px; text-decoration: none; text-transform: none;" href="{url}" target="_blank" rel="noopener"> <strong style="color: white; font-weight: bold; text: white;">プロジェクトを開く</strong> </a></span></p>
                        <p>ありがとう</p>
                        <p>{company_name}</p>
                        <p><span style="color: #000000; font-family: "Open Sans", sans-serif; font-size: 14px; background-color: #ffffff;"><strong>{app_name}</strong></span></p>',

                        'nl' => '<p><strong>Hallo,{name} </strong></p>
                        <p>Je wordt uitgenodigd voor een nieuw project {project} door {project_creater_name} .</p>
                        <p style="text-align: left;" align="center">Neem gerust contact op als u vragen heeft.</p>
                        <p style="text-align: center;" align="center"><span style="font-size: 18pt;"><a style="background: #6676ef; color: #ffffff; font-family: "Open Sans", Helvetica, Arial, sans-serif; font-weight: normal; line-height: 120%; margin: 0px; text-decoration: none; text-transform: none;" href="{url}" target="_blank" rel="noopener"> <strong style="color: white; font-weight: bold; text: white;">Project openen</strong> </a></span></p>
                        <p>Bedankt</p>
                        <p>{company_name}</p>
                        <p><span style="color: #000000; font-family: "Open Sans", sans-serif; font-size: 14px; background-color: #ffffff;"><strong>{app_name}</strong></span></p>',

                        'pl' => '<p><strong>Cześć,{name} </strong></p>
                        <p>Zapraszamy do nowego projektu {project} przez {project_creater_name} .</p>
                        <p style="text-align: left;" align="center">Jeśli masz jakiekolwiek pytania, skontaktuj się z nami.</p>
                        <p style="text-align: center;" align="center"><span style="font-size: 18pt;"><a style="background: #6676ef; color: #ffffff; font-family: "Open Sans", Helvetica, Arial, sans-serif; font-weight: normal; line-height: 120%; margin: 0px; text-decoration: none; text-transform: none;" href="{url}" target="_blank" rel="noopener"> <strong style="color: white; font-weight: bold; text: white;">Otwórz projekt</strong> </a></span></p>
                        <p>Dziękuję</p>
                        <p>{company_name}</p>
                        <p><span style="color: #000000; font-family: "Open Sans", sans-serif; font-size: 14px; background-color: #ffffff;"><strong>{app_name}</strong></span></p>',

                        'pt' => '<p><strong>Olá,{name} </strong></p>
                        <p>Você está convidado para um novo projeto {project} por {project_creater_name} .</p>
                        <p style="text-align: left;" align="center">Sinta-se à vontade para entrar em contato se tiver alguma dúvida.</p>
                        <p style="text-align: center;" align="center"><span style="font-size: 18pt;"><a style="background: #6676ef; color: #ffffff; font-family: "Open Sans", Helvetica, Arial, sans-serif; font-weight: normal; line-height: 120%; margin: 0px; text-decoration: none; text-transform: none;" href="{url}" target="_blank" rel="noopener"> <strong style="color: white; font-weight: bold; text: white;">Abrir projeto</strong> </a></span></p>
                        <p>Obrigado</p>
                        <p>{company_name}</p>
                        <p><span style="color: #000000; font-family: "Open Sans", sans-serif; font-size: 14px; background-color: #ffffff;"><strong>{app_name}</strong></span></p>',
                    
                        'ru' => '<p><strong>Привет,{name} </strong></p>
                        <p>Вас приглашают в новый проект {project} by {project_creater_name} .</p>
                        <p style="text-align: left;" align="center">Не стесняйтесь обращаться, если у вас есть какие-либо вопросы.</p>
                        <p style="text-align: center;" align="center"><span style="font-size: 18pt;"><a style="background: #6676ef; color: #ffffff; font-family: "Open Sans", Helvetica, Arial, sans-serif; font-weight: normal; line-height: 120%; margin: 0px; text-decoration: none; text-transform: none;" href="{url}" target="_blank" rel="noopener"> <strong style="color: white; font-weight: bold; text: white;">Открытый проект</strong> </a></span></p>
                        <p>Спасибо</p>
                        <p>{company_name}</p>
                        <p><span style="color: #000000; font-family: "Open Sans", sans-serif; font-size: 14px; background-color: #ffffff;"><strong>{app_name}</strong></span></p>',

                        'tr' => '<p><strong>Merhaba,{name} </strong></p>
                        <p>Yeni projeye davetlisiniz {project} ile {project_creater_name} .</p>
                        <p style="text-align: left;" align="center">Herhangi bir sorunuz varsa bizimle iletişime geçmekten çekinmeyin.</p>
                        <p style="text-align: center;" align="center"><span style="font-size: 18pt;"><a style="background: #6676ef; color: #ffffff; font-family: "Open Sans", Helvetica, Arial, sans-serif; font-weight: normal; line-height: 120%; margin: 0px; text-decoration: none; text-transform: none;" href="{url}" target="_blank" rel="noopener"> <strong style="color: white; font-weight: bold; text: white;">Projeyi Aç</strong> </a></span></p>
                        <p>Teşekkür ederim</p>
                        <p>{company_name}</p>
                        <p><span style="color: #000000; font-family: "Open Sans", sans-serif; font-size: 14px; background-color: #ffffff;"><strong>{app_name}</strong></span></p>',
                    ],
            ],

            'Project Assigned' => [
                'subject' =>'New Project Share',
                'variables' => '{
                    "App Url": "app_url",
                    "App Name": "app_name",
                    "Company Name": "company_name",
                    "Project Creater Name": "project_creater_name",
                    "User": "name",
                    "Project": "project",
                    "URL": "url"
                    
                  }',


                  'lang' => [

                    'ar' => '<p><strong>مرحبًا,{name} </strong></p>
                    <p>أنت مدعو إلى مشروع جديد {project} بواسطة {project_creater_name} .</p>
                    <p style="text-align: left;" align="center">لا تتردد في التواصل معنا إذا كان لديك أي أسئلة.</p>
                    <p style="text-align: center;" align="center"><span style="font-size: 18pt;"><a style="background: #6676ef; color: #ffffff; font-family: "Open Sans", Helvetica, Arial, sans-serif; font-weight: normal; line-height: 120%; margin: 0px; text-decoration: none; text-transform: none;" href="{url}" target="_blank" rel="noopener"> <strong style="color: white; font-weight: bold; text: white;">افتح المشروع</strong> </a></span></p>
                    <p>شكرًا لك</p>
                    <p>{company_name}</p>
                    <p><span style="color: #000000; font-family: "Open Sans", sans-serif; font-size: 14px; background-color: #ffffff;"><strong>{app_name}</strong></span></p>',

                    'da' => '<p><strong>Hej,{name} </strong></p>
                    <p>Du inviteres ind i nyt projekt {project} ved {project_creater_name} .</p>
                    <p style="text-align: left;" align="center">Du er velkommen til at kontakte os, hvis du har spørgsmål.</p>
                    <p style="text-align: center;" align="center"><span style="font-size: 18pt;"><a style="background: #6676ef; color: #ffffff; font-family: "Open Sans", Helvetica, Arial, sans-serif; font-weight: normal; line-height: 120%; margin: 0px; text-decoration: none; text-transform: none;" href="{url}" target="_blank" rel="noopener"> <strong style="color: white; font-weight: bold; text: white;">Åbn projekt</strong> </a></span></p>
                    <p>Tak</p>
                    <p>{company_name}</p>
                    <p><span style="color: #000000; font-family: "Open Sans", sans-serif; font-size: 14px; background-color: #ffffff;"><strong>{app_name}</strong></span></p>',

                    'de' => '<p><strong>Hallo,{name} </strong></p>
                    <p>Sie werden zu einem neuen Projekt eingeladen {project} von {project_creater_name} .</p>
                    <p style="text-align: left;" align="center">Wenn Sie Fragen haben, können Sie sich jederzeit an uns wenden.</p>
                    <p style="text-align: center;" align="center"><span style="font-size: 18pt;"><a style="background: #6676ef; color: #ffffff; font-family: "Open Sans", Helvetica, Arial, sans-serif; font-weight: normal; line-height: 120%; margin: 0px; text-decoration: none; text-transform: none;" href="{url}" target="_blank" rel="noopener"> <strong style="color: white; font-weight: bold; text: white;">Projekt öffnen</strong> </a></span></p>
                    <p>Danke</p>
                    <p>{company_name}</p>
                    <p><span style="color: #000000; font-family: "Open Sans", sans-serif; font-size: 14px; background-color: #ffffff;"><strong>{app_name}</strong></span></p>',

                    'en' => '<p><strong>Hello,{name} </strong></p>
                    <p>You are invited into new project {project} by {project_creater_name} .</p>
                    <p style="text-align: left;" align="center">Feel free to reach out if you have any questions.</p>
                    <p style="text-align: center;" align="center"><span style="font-size: 18pt;"><a style="background: #6676ef; color: #ffffff; font-family: "Open Sans", Helvetica, Arial, sans-serif; font-weight: normal; line-height: 120%; margin: 0px; text-decoration: none; text-transform: none;" href="{url}" target="_blank" rel="noopener"> <strong style="color: white; font-weight: bold; text: white;">Open Project</strong> </a></span></p>
                    <p>Thank you</p>
                    <p>{company_name}</p>
                    <p><span style="color: #000000; font-family: "Open Sans", sans-serif; font-size: 14px; background-color: #ffffff;"><strong>{app_name}</strong></span></p>',

                    'es' => '<p><strong>Hola,{name} </strong></p>
                    <p>Estás invitado a un nuevo proyecto. {project} por {project_creater_name} .</p>
                    <p style="text-align: left;" align="center">No dude en comunicarse si tiene alguna pregunta.</p>
                    <p style="text-align: center;" align="center"><span style="font-size: 18pt;"><a style="background: #6676ef; color: #ffffff; font-family: "Open Sans", Helvetica, Arial, sans-serif; font-weight: normal; line-height: 120%; margin: 0px; text-decoration: none; text-transform: none;" href="{url}" target="_blank" rel="noopener"> <strong style="color: white; font-weight: bold; text: white;">Abrir proyecto</strong> </a></span></p>
                    <p>Gracias</p>
                    <p>{company_name}</p>
                    <p><span style="color: #000000; font-family: "Open Sans", sans-serif; font-size: 14px; background-color: #ffffff;"><strong>{app_name}</strong></span></p>',

                    'fr' => '<p><strong>Bonjour,{name} </strong></p>
                    <p>Vous êtes invité dans un nouveau projet {project} par {project_creater_name} .</p>
                    <p style="text-align: left;" align="center">N hésitez pas à nous contacter si vous avez des questions.</p>
                    <p style="text-align: center;" align="center"><span style="font-size: 18pt;"><a style="background: #6676ef; color: #ffffff; font-family: "Open Sans", Helvetica, Arial, sans-serif; font-weight: normal; line-height: 120%; margin: 0px; text-decoration: none; text-transform: none;" href="{url}" target="_blank" rel="noopener"> <strong style="color: white; font-weight: bold; text: white;">Projet ouvert</strong> </a></span></p>
                    <p>Merci</p>
                    <p>{company_name}</p>
                    <p><span style="color: #000000; font-family: "Open Sans", sans-serif; font-size: 14px; background-color: #ffffff;"><strong>{app_name}</strong></span></p>',

                    'it' => '<p><strong>Ciao,{name} </strong></p>
                    <p>Sei invitato a un nuovo progetto {project} di {project_creater_name} .</p>
                    <p style="text-align: left;" align="center">Sentiti libero di contattarci se hai domande.</p>
                    <p style="text-align: center;" align="center"><span style="font-size: 18pt;"><a style="background: #6676ef; color: #ffffff; font-family: "Open Sans", Helvetica, Arial, sans-serif; font-weight: normal; line-height: 120%; margin: 0px; text-decoration: none; text-transform: none;" href="{url}" target="_blank" rel="noopener"> <strong style="color: white; font-weight: bold; text: white;">Apri progetto</strong> </a></span></p>
                    <p>Grazie</p>
                    <p>{company_name}</p>
                    <p><span style="color: #000000; font-family: "Open Sans", sans-serif; font-size: 14px; background-color: #ffffff;"><strong>{app_name}</strong></span></p>',

                    'ja' => '<p><strong>こんにちは,{name} </strong></p>
                    <p>新しいプロジェクトに招待されています {project} による {project_creater_name} .</p>
                    <p style="text-align: left;" align="center">ご質問がございましたら、お気軽にお問い合わせください。</p>
                    <p style="text-align: center;" align="center"><span style="font-size: 18pt;"><a style="background: #6676ef; color: #ffffff; font-family: "Open Sans", Helvetica, Arial, sans-serif; font-weight: normal; line-height: 120%; margin: 0px; text-decoration: none; text-transform: none;" href="{url}" target="_blank" rel="noopener"> <strong style="color: white; font-weight: bold; text: white;">プロジェクトを開く</strong> </a></span></p>
                    <p>ありがとう</p>
                    <p>{company_name}</p>
                    <p><span style="color: #000000; font-family: "Open Sans", sans-serif; font-size: 14px; background-color: #ffffff;"><strong>{app_name}</strong></span></p>',

                    'nl' => '<p><strong>Hallo,{name} </strong></p>
                    <p>Je wordt uitgenodigd voor een nieuw project {project} door {project_creater_name} .</p>
                    <p style="text-align: left;" align="center">Neem gerust contact op als u vragen heeft.</p>
                    <p style="text-align: center;" align="center"><span style="font-size: 18pt;"><a style="background: #6676ef; color: #ffffff; font-family: "Open Sans", Helvetica, Arial, sans-serif; font-weight: normal; line-height: 120%; margin: 0px; text-decoration: none; text-transform: none;" href="{url}" target="_blank" rel="noopener"> <strong style="color: white; font-weight: bold; text: white;">Project openen</strong> </a></span></p>
                    <p>Bedankt</p>
                    <p>{company_name}</p>
                    <p><span style="color: #000000; font-family: "Open Sans", sans-serif; font-size: 14px; background-color: #ffffff;"><strong>{app_name}</strong></span></p>',

                    'pl' => '<p><strong>Cześć,{name} </strong></p>
                    <p>Zapraszamy do nowego projektu {project} przez {project_creater_name} .</p>
                    <p style="text-align: left;" align="center">Jeśli masz jakiekolwiek pytania, skontaktuj się z nami.</p>
                    <p style="text-align: center;" align="center"><span style="font-size: 18pt;"><a style="background: #6676ef; color: #ffffff; font-family: "Open Sans", Helvetica, Arial, sans-serif; font-weight: normal; line-height: 120%; margin: 0px; text-decoration: none; text-transform: none;" href="{url}" target="_blank" rel="noopener"> <strong style="color: white; font-weight: bold; text: white;">Otwórz projekt</strong> </a></span></p>
                    <p>Dziękuję</p>
                    <p>{company_name}</p>
                    <p><span style="color: #000000; font-family: "Open Sans", sans-serif; font-size: 14px; background-color: #ffffff;"><strong>{app_name}</strong></span></p>',

                    'pt' => '<p><strong>Olá,{name} </strong></p>
                    <p>Você está convidado para um novo projeto {project} por {project_creater_name} .</p>
                    <p style="text-align: left;" align="center">Sinta-se à vontade para entrar em contato se tiver alguma dúvida.</p>
                    <p style="text-align: center;" align="center"><span style="font-size: 18pt;"><a style="background: #6676ef; color: #ffffff; font-family: "Open Sans", Helvetica, Arial, sans-serif; font-weight: normal; line-height: 120%; margin: 0px; text-decoration: none; text-transform: none;" href="{url}" target="_blank" rel="noopener"> <strong style="color: white; font-weight: bold; text: white;">Abrir projeto</strong> </a></span></p>
                    <p>Obrigado</p>
                    <p>{company_name}</p>
                    <p><span style="color: #000000; font-family: "Open Sans", sans-serif; font-size: 14px; background-color: #ffffff;"><strong>{app_name}</strong></span></p>',
                
                    'ru' => '<p><strong>Привет,{name} </strong></p>
                    <p>Вас приглашают в новый проект {project} by {project_creater_name} .</p>
                    <p style="text-align: left;" align="center">Не стесняйтесь обращаться, если у вас есть какие-либо вопросы.</p>
                    <p style="text-align: center;" align="center"><span style="font-size: 18pt;"><a style="background: #6676ef; color: #ffffff; font-family: "Open Sans", Helvetica, Arial, sans-serif; font-weight: normal; line-height: 120%; margin: 0px; text-decoration: none; text-transform: none;" href="{url}" target="_blank" rel="noopener"> <strong style="color: white; font-weight: bold; text: white;">Открытый проект</strong> </a></span></p>
                    <p>Спасибо</p>
                    <p>{company_name}</p>
                    <p><span style="color: #000000; font-family: "Open Sans", sans-serif; font-size: 14px; background-color: #ffffff;"><strong>{app_name}</strong></span></p>',

                    'tr' => '<p><strong>Merhaba,{name} </strong></p>
                    <p>Yeni projeye davetlisiniz {project} ile {project_creater_name} .</p>
                    <p style="text-align: left;" align="center">Herhangi bir sorunuz varsa bizimle iletişime geçmekten çekinmeyin.</p>
                    <p style="text-align: center;" align="center"><span style="font-size: 18pt;"><a style="background: #6676ef; color: #ffffff; font-family: "Open Sans", Helvetica, Arial, sans-serif; font-weight: normal; line-height: 120%; margin: 0px; text-decoration: none; text-transform: none;" href="{url}" target="_blank" rel="noopener"> <strong style="color: white; font-weight: bold; text: white;">Projeyi Aç</strong> </a></span></p>
                    <p>Teşekkür ederim</p>
                    <p>{company_name}</p>
                    <p><span style="color: #000000; font-family: "Open Sans", sans-serif; font-size: 14px; background-color: #ffffff;"><strong>{app_name}</strong></span></p>',
                ],
            ],

        ];

        foreach ($emailTemplate as $eTemp) {
            $table = EmailTemplate::where('name', $eTemp)->where('module_name', 'Taskly')->exists();
            if (!$table) {
                $emailtemplate =  EmailTemplate::create(
                    [
                        'name' => $eTemp,
                        'from' => 'Taskly',
                        'module_name' => 'Taskly',
                        'created_by' => 1,
                        'workspace_id' => 0
                    ]
                );
                foreach ($defaultTemplate[$eTemp]['lang'] as $lang => $content) {
                    EmailTemplateLang::create(
                        [
                            'parent_id' => $emailtemplate->id,
                            'lang' => $lang,
                            'subject' => $defaultTemplate[$eTemp]['subject'],
                            'variables' => $defaultTemplate[$eTemp]['variables'],
                            'content' => $content,
                        ]
                    );
                }
            }
        }
    }
}

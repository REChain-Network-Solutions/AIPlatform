import { defineConfig } from 'vite';
import laravel from 'laravel-vite-plugin';
import { readdirSync, statSync } from 'fs';
import { join,relative,dirname } from 'path';
import { fileURLToPath } from 'url';

export default defineConfig({
    build: {
        outDir: '../../public/build-pmcore',
        emptyOutDir: true,
        manifest: true,
    },
    plugins: [
        laravel({
            publicDirectory: '../../public',
            buildDirectory: 'build-pmcore',
            input: [
                __dirname + '/resources/assets/sass/app.scss',
                __dirname + '/resources/assets/js/app.js',
                __dirname + '/resources/assets/js/dashboard.js',
                __dirname + '/resources/assets/js/project-show.js',
                __dirname + '/resources/assets/js/project-tasks-board.js',
                __dirname + '/resources/assets/js/project-tasks.js',
                __dirname + '/resources/assets/js/task-form-handler.js',
                __dirname + '/resources/assets/js/project-list.js',
                __dirname + '/resources/assets/js/project-form.js',
                __dirname + '/resources/assets/js/project-statuses.js',
                __dirname + '/resources/assets/js/timesheets.js',
                __dirname + '/resources/assets/js/timesheets-form.js',
                __dirname + '/resources/assets/js/timesheets-edit.js'
            ],
            refresh: true,
        }),
    ],
});
// Scen all resources for assets file. Return array
//function getFilePaths(dir) {
//    const filePaths = [];
//
//    function walkDirectory(currentPath) {
//        const files = readdirSync(currentPath);
//        for (const file of files) {
//            const filePath = join(currentPath, file);
//            const stats = statSync(filePath);
//            if (stats.isFile() && !file.startsWith('.')) {
//                const relativePath = 'Modules/PMCore/'+relative(__dirname, filePath);
//                filePaths.push(relativePath);
//            } else if (stats.isDirectory()) {
//                walkDirectory(filePath);
//            }
//        }
//    }
//
//    walkDirectory(dir);
//    return filePaths;
//}

//const __filename = fileURLToPath(import.meta.url);
//const __dirname = dirname(__filename);

//const assetsDir = join(__dirname, 'resources/assets');
//export const paths = getFilePaths(assetsDir);


// Import paths from paths.js
import { paths } from './paths.js';
export { paths };

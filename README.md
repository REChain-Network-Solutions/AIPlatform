<html>
<body>
<!--StartFragment--><html><head></head><body><h1>MagicAI (AIPlatform) Wiki</h1>
<hr>
<h2>🌐 Overview</h2>
<ul>
<li>
<p><strong>Repository</strong>: <a href="https://github.com/REChain-Network-Solutions/AIPlatform.git">REChain-Network-Solutions/AIPlatform</a></p>
</li>
<li>
<p><strong>Codename / Product</strong>: MagicAI</p>
</li>
<li>
<p><strong>Purpose</strong>: AI SaaS starter platform including landing page, dashboard, customization, and core AI features.</p>
</li>
<li>
<p><strong>License</strong>: BSD 3-Clause</p>
</li>
<li>
<p><strong>Languages</strong>: JavaScript (41.4%), PHP (27%), Blade (20%), CSS/SCSS/Sass (remaining)</p>
</li>
</ul>
<hr>
<h2>🚀 Key Features</h2>
<ul>
<li>
<p><strong>Landing Page &amp; Dashboard</strong>: Out-of-the-box templates for SaaS presentation and operations.</p>
</li>
<li>
<p><strong>Theme Customization</strong>: Change visual appearance with one click.</p>
</li>
<li>
<p><strong>Extensions Support</strong>: Modular AI features via <code inline="">magicai-extentions</code>.</p>
</li>
<li>
<p><strong>Localization</strong>: Multi-language support via <code inline="">lang/</code>.</p>
</li>
<li>
<p><strong>Secure Foundation</strong>: Security policy included in <code inline="">SECURITY.md</code>.</p>
</li>
</ul>
<hr>
<h2>📂 Repository Structure</h2>

Path | Description
-- | --
.github/ | GitHub workflows, issue templates
app/ | Core application logic: controllers, models, services
bootstrap/ | Initialization & environment bootstrapping
config/ | Configuration files (app, DB, cache, etc.)
database/ | Migrations, seeds, SQL dump (magicai.sql)
documents/ | Documentation & project notes
lang/ | Translation & localization files
magicai-extentions/ | Custom AI extensions/plugins
packages/ | External or internal packages/modules
public/ | Public assets (CSS, JS, images)
resources/ | Views, Blade templates, frontend resources
routes/ | API & web routes
storage/ | Logs, cached data, temporary files
tests/ | Unit & integration tests


<p>Other important files:</p>
<ul>
<li>
<p><code inline="">.env.example</code> → environment template</p>
</li>
<li>
<p><code inline="">composer.json</code> / <code inline="">composer.lock</code> → PHP dependencies</p>
</li>
<li>
<p><code inline="">package.json</code> / <code inline="">package-lock.json</code> → JavaScript dependencies</p>
</li>
<li>
<p><code inline="">vite.config.mjs</code> → Vite frontend bundler config</p>
</li>
<li>
<p><code inline="">phpunit.xml</code> → PHPUnit test configuration</p>
</li>
<li>
<p><code inline="">version.txt</code> → Current app version</p>
</li>
<li>
<p><code inline="">README.md</code>, <code inline="">SECURITY.md</code> → Documentation</p>
</li>
</ul>
<hr>
<h2>⚙️ Installation &amp; Setup</h2>
<ol>
<li>
<p><strong>Clone Repository</strong></p>
<pre><code class="language-bash">git clone https://github.com/REChain-Network-Solutions/AIPlatform.git
cd AIPlatform
</code></pre>
</li>
<li>
<p><strong>Environment Setup</strong></p>
<pre><code class="language-bash">cp .env.example .env
# edit DB credentials, API keys, etc.
</code></pre>
</li>
<li>
<p><strong>Install Dependencies</strong></p>
<pre><code class="language-bash">composer install
npm install   # or yarn
</code></pre>
</li>
<li>
<p><strong>Database Setup</strong></p>
<pre><code class="language-bash">php artisan migrate --seed
# optionally import magicai.sql
</code></pre>
</li>
<li>
<p><strong>Build Frontend</strong></p>
<pre><code class="language-bash">npm run build   # or npm run dev for development
</code></pre>
</li>
<li>
<p><strong>Run Application</strong></p>
<pre><code class="language-bash">php artisan serve
</code></pre>
</li>
</ol>
<hr>
<h2>🧩 Architecture</h2>
<ul>
<li>
<p><strong>Backend</strong>: Laravel-based (PHP + Blade templating)</p>
</li>
<li>
<p><strong>Frontend</strong>: Tailwind CSS + Vite build system</p>
</li>
<li>
<p><strong>Database</strong>: SQL migrations &amp; schema (<code inline="">magicai.sql</code>)</p>
</li>
<li>
<p><strong>Extensions</strong>: Modular plug-ins via <code inline="">magicai-extentions</code></p>
</li>
<li>
<p><strong>Localization</strong>: Multi-language system in <code inline="">lang/</code></p>
</li>
<li>
<p><strong>Testing</strong>: PHPUnit for backend, potential JS tests</p>
</li>
</ul>
<hr>
<h2>🔐 Security</h2>
<ul>
<li>
<p>Security guidelines in <code inline="">SECURITY.md</code></p>
</li>
<li>
<p>BSD 3-Clause license</p>
</li>
<li>
<p><code inline="">.env</code> file for sensitive configs (DB, API keys)</p>
</li>
<li>
<p>Recommend HTTPS + reverse proxy for production</p>
</li>
</ul>
<hr>
<h2>📊 Project Status</h2>
<ul>
<li>
<p><strong>Commits</strong>: ~7 (early development stage)</p>
</li>
<li>
<p><strong>Releases</strong>: none published</p>
</li>
<li>
<p><strong>Stars/Forks</strong>: minimal (early community adoption)</p>
</li>
</ul>
<hr>
<h2>🌍 Usage Scenarios</h2>
<ul>
<li>
<p>SaaS startups needing ready AI landing + dashboard</p>
</li>
<li>
<p>Enterprises requiring multilingual AI interface</p>
</li>
<li>
<p>Developers launching branded AI tools with quick theme customization</p>
</li>
<li>
<p>Educational projects experimenting with AI SaaS architecture</p>
</li>
</ul>
<hr>
<h2>📌 Roadmap (Suggested)</h2>
<ul class="contains-task-list">
<li class="task-list-item">
<p><input type="checkbox" disabled=""> Publish first official release</p>
</li>
<li class="task-list-item">
<p><input type="checkbox" disabled=""> Extend AI integrations (OpenAI, HuggingFace, local models)</p>
</li>
<li class="task-list-item">
<p><input type="checkbox" disabled=""> Add CI/CD pipelines in <code inline="">.github/workflows/</code></p>
</li>
<li class="task-list-item">
<p><input type="checkbox" disabled=""> Dockerize for production deployment</p>
</li>
<li class="task-list-item">
<p><input type="checkbox" disabled=""> Expand automated test coverage</p>
</li>
<li class="task-list-item">
<p><input type="checkbox" disabled=""> Create official documentation site</p>
</li>
</ul>
<hr>
<h2>🧭 Summary</h2>
<p>MagicAI (AIPlatform) is a <strong>flexible AI SaaS starter kit</strong> offering a complete landing + dashboard solution, fast theming, localization, and extensibility. Though early in development, it provides a strong base for building scalable AI-powered applications.</p></body></html><!--EndFragment-->
</body>
</html># MagicAI (AIPlatform) Wiki

---

## 🌐 Overview

* **Repository**: [[REChain-Network-Solutions/AIPlatform](https://github.com/REChain-Network-Solutions/AIPlatform.git)](https://github.com/REChain-Network-Solutions/AIPlatform.git)
* **Codename / Product**: MagicAI
* **Purpose**: AI SaaS starter platform including landing page, dashboard, customization, and core AI features.
* **License**: BSD 3-Clause
* **Languages**: JavaScript (41.4%), PHP (27%), Blade (20%), CSS/SCSS/Sass (remaining)

---

## 🚀 Key Features

* **Landing Page & Dashboard**: Out-of-the-box templates for SaaS presentation and operations.
* **Theme Customization**: Change visual appearance with one click.
* **Extensions Support**: Modular AI features via `magicai-extentions`.
* **Localization**: Multi-language support via `lang/`.
* **Secure Foundation**: Security policy included in `SECURITY.md`.

---

## 📂 Repository Structure

| Path                  | Description                                           |
| --------------------- | ----------------------------------------------------- |
| `.github/`            | GitHub workflows, issue templates                     |
| `app/`                | Core application logic: controllers, models, services |
| `bootstrap/`          | Initialization & environment bootstrapping            |
| `config/`             | Configuration files (app, DB, cache, etc.)            |
| `database/`           | Migrations, seeds, SQL dump (`magicai.sql`)           |
| `documents/`          | Documentation & project notes                         |
| `lang/`               | Translation & localization files                      |
| `magicai-extentions/` | Custom AI extensions/plugins                          |
| `packages/`           | External or internal packages/modules                 |
| `public/`             | Public assets (CSS, JS, images)                       |
| `resources/`          | Views, Blade templates, frontend resources            |
| `routes/`             | API & web routes                                      |
| `storage/`            | Logs, cached data, temporary files                    |
| `tests/`              | Unit & integration tests                              |

Other important files:

* `.env.example` → environment template
* `composer.json` / `composer.lock` → PHP dependencies
* `package.json` / `package-lock.json` → JavaScript dependencies
* `vite.config.mjs` → Vite frontend bundler config
* `phpunit.xml` → PHPUnit test configuration
* `version.txt` → Current app version
* `README.md`, `SECURITY.md` → Documentation

---

## ⚙️ Installation & Setup

1. **Clone Repository**

   ```bash
   git clone https://github.com/REChain-Network-Solutions/AIPlatform.git
   cd AIPlatform
   ```

2. **Environment Setup**

   ```bash
   cp .env.example .env
   # edit DB credentials, API keys, etc.
   ```

3. **Install Dependencies**

   ```bash
   composer install
   npm install   # or yarn
   ```

4. **Database Setup**

   ```bash
   php artisan migrate --seed
   # optionally import magicai.sql
   ```

5. **Build Frontend**

   ```bash
   npm run build   # or npm run dev for development
   ```

6. **Run Application**

   ```bash
   php artisan serve
   ```

---

## 🧩 Architecture

* **Backend**: Laravel-based (PHP + Blade templating)
* **Frontend**: Tailwind CSS + Vite build system
* **Database**: SQL migrations & schema (`magicai.sql`)
* **Extensions**: Modular plug-ins via `magicai-extentions`
* **Localization**: Multi-language system in `lang/`
* **Testing**: PHPUnit for backend, potential JS tests

---

## 🔐 Security

* Security guidelines in `SECURITY.md`
* BSD 3-Clause license
* `.env` file for sensitive configs (DB, API keys)
* Recommend HTTPS + reverse proxy for production

---

## 📊 Project Status

* **Commits**: \~7 (early development stage)
* **Releases**: none published
* **Stars/Forks**: minimal (early community adoption)

---

## 🌍 Usage Scenarios

* SaaS startups needing ready AI landing + dashboard
* Enterprises requiring multilingual AI interface
* Developers launching branded AI tools with quick theme customization
* Educational projects experimenting with AI SaaS architecture

---

## 📌 Roadmap (Suggested)

* [ ] Publish first official release
* [ ] Extend AI integrations (OpenAI, HuggingFace, local models)
* [ ] Add CI/CD pipelines in `.github/workflows/`
* [ ] Dockerize for production deployment
* [ ] Expand automated test coverage
* [ ] Create official documentation site

---

## 🧭 Summary

MagicAI (AIPlatform) is a **flexible AI SaaS starter kit** offering a complete landing + dashboard solution, fast theming, localization, and extensibility. Though early in development, it provides a strong base for building scalable AI-powered applications.

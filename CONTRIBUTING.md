# Contributing to AIPlatform

Thank you for your interest in contributing to AIPlatform! We appreciate your time and effort in helping us build a better platform.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Code Style](#code-style)
- [Commit Message Guidelines](#commit-message-guidelines)
- [Pull Request Process](#pull-request-process)
- [Reporting Issues](#reporting-issues)
- [Feature Requests](#feature-requests)
- [License](#license)

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally
   ```bash
   git clone git@github.com:your-username/AIPlatform.git
   cd AIPlatform
   ```
3. **Set up the development environment**
   ```bash
   cp .env.example .env
   composer install
   npm install
   php artisan key:generate
   ```
4. **Create a branch** for your feature or bugfix
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b bugfix/issue-number-short-description
   ```

## Development Workflow

1. **Sync with main**
   ```bash
   git fetch upstream
   git merge upstream/main
   ```
2. **Make your changes** following the code style guidelines
3. **Run tests** to ensure nothing is broken
   ```bash
   composer test
   ```
4. **Commit your changes** with a descriptive commit message
5. **Push to your fork**
   ```bash
   git push origin your-branch-name
   ```
6. **Open a Pull Request** against the `main` branch

## Code Style

We follow [PSR-12](https://www.php-fig.org/psr/psr-12/) for PHP code and [Airbnb JavaScript Style Guide](https://github.com/airbnb/javascript) for JavaScript.

### PHP

We use [Laravel Pint](https://laravel.com/docs/pint) for PHP code style. To fix style issues:

```bash
composer pint
```

### JavaScript

We use [ESLint](https://eslint.org/) and [Prettier](https://prettier.io/) for JavaScript code style:

```bash
npm run lint
# Fix auto-fixable issues
npm run lint:fix
```

## Commit Message Guidelines

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

### Types

- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation only changes
- `style`: Changes that do not affect the meaning of the code (white-space, formatting, etc.)
- `refactor`: A code change that neither fixes a bug nor adds a feature
- `perf`: A code change that improves performance
- `test`: Adding missing or correcting existing tests
- `chore`: Changes to the build process or auxiliary tools and libraries

### Examples

```
feat(auth): add two-factor authentication

description of the change

Closes #123
```

## Pull Request Process

1. Ensure your code follows the style guidelines
2. Update the documentation if needed
3. Add tests for your changes
4. Ensure all tests pass
5. Update the CHANGELOG.md with your changes
6. Open a Pull Request with a clear title and description

## Reporting Issues

When reporting issues, please include:

1. A clear title and description
2. Steps to reproduce the issue
3. Expected and actual behavior
4. Screenshots if applicable
5. Your environment (OS, PHP version, etc.)

## Feature Requests

For feature requests, please:

1. Check if the feature has already been requested
2. Clearly describe the feature and why it would be useful
3. Include any relevant use cases or examples

## License

By contributing to AIPlatform, you agree that your contributions will be licensed under the [MIT License](LICENSE).

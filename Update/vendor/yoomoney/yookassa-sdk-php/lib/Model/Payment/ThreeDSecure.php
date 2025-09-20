<?php

/*
 * The MIT License
 *
 * Copyright (c) 2025 "YooMoney", NBСO LLC
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

namespace YooKassa\Model\Payment;

use Exception;
use YooKassa\Common\AbstractObject;
use YooKassa\Validator\Constraints as Assert;

/**
 * Класс, представляющий модель ThreeDSecure.
 *
 * Данные о прохождении пользователем аутентификации по 3‑D Secure для подтверждения платежа.
 *
 * @category Class
 * @package  YooKassa\Model
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @property bool $applied Отображение пользователю формы для прохождения аутентификации по 3‑D Secure.
 */
class ThreeDSecure extends AbstractObject
{
    /**
     * Отображение пользователю формы для прохождения аутентификации по 3‑D Secure.
     *
     * Возможные значения:
     * - `true` — ЮKassa отобразила пользователю форму, чтобы он мог пройти аутентификацию по 3‑D Secure;
     * - `false` — платеж проходил без аутентификации по 3‑D Secure.
     *
     * @var bool
     */
    #[Assert\NotNull]
    #[Assert\Type('bool')]
    private ?bool $_applied = false;

    /**
     * Возвращает applied.
     *
     * @return bool|null
     */
    public function getApplied(): ?bool
    {
        return $this->_applied;
    }

    /**
     * Устанавливает applied.
     *
     * @param bool|null $applied Отображение пользователю формы для прохождения аутентификации по 3‑D Secure. Возможные значения:  * ~`true` — ЮKassa отобразила пользователю форму, чтобы он мог пройти аутентификацию по 3‑D Secure; * ~`false` — платеж проходил без аутентификации по 3‑D Secure.
     *
     * @return self
     */
    public function setApplied(?bool $applied = null): self
    {
        $this->_applied = $this->validatePropertyValue('_applied', $applied);
        return $this;
    }

}

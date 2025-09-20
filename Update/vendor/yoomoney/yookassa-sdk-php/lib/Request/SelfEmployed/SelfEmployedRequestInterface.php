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

namespace YooKassa\Request\SelfEmployed;


use YooKassa\Model\SelfEmployed\SelfEmployedConfirmation;

/**
 * Interface SelfEmployedRequestInterface.
 *
 * Запрос на создание объекта самозанятого.
 *
 * @category Interface
 * @package  YooKassa\Request
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 * @property null|string $itn ИНН самозанятого.
 * @property null|string $phone Телефон самозанятого, который привязан к личному кабинету в сервисе Мой налог.
 * @property null|SelfEmployedRequestConfirmation $confirmation Сценарий подтверждения пользователем заявки ЮMoney на получение прав для регистрации чеков в сервисе Мой налог.
 */
interface SelfEmployedRequestInterface
{
    /**
     * Возвращает ИНН самозанятого.
     *
     * @return null|string ИНН самозанятого
     */
    public function getItn(): ?string;

    /**
     * Устанавливает ИНН самозанятого.
     *
     * @param null|string $itn ИНН самозанятого
     *
     * @return $this
     */
    public function setItn(?string $itn = null): self;

    /**
     * Проверяет наличие ИНН самозанятого в запросе.
     *
     * @return bool True если ИНН самозанятого задан, false если нет
     */
    public function hasItn(): bool;

    /**
     * Возвращает телефон самозанятого.
     *
     * @return null|string Телефон самозанятого
     */
    public function getPhone(): ?string;

    /**
     * Устанавливает телефон самозанятого.
     *
     * @param null|string $phone Телефон самозанятого
     *
     * @return $this
     */
    public function setPhone(?string $phone = null): self;

    /**
     * Проверяет наличие телефона самозанятого в запросе.
     *
     * @return bool True если телефон самозанятого задан, false если нет
     */
    public function hasPhone(): bool;

    /**
     * Возвращает сценарий подтверждения.
     */
    public function getConfirmation(): ?SelfEmployedRequestConfirmation;

    /**
     * Устанавливает сценарий подтверждения.
     *
     * @param null|array|SelfEmployedConfirmation $confirmation Сценарий подтверждения
     *
     * @return $this
     */
    public function setConfirmation(mixed $confirmation = null): self;

    /**
     * Проверяет наличие сценария подтверждения самозанятого в запросе.
     *
     * @return bool True если сценарий подтверждения самозанятого задан, false если нет
     */
    public function hasConfirmation(): bool;

    /**
     * Проверяет на валидность текущий объект
     *
     * @return bool True если объект запроса валиден, false если нет
     */
    public function validate(): bool;
}

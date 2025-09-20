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

namespace YooKassa\Model\SelfEmployed;

use DateTime;
use Exception;
use YooKassa\Common\Exceptions\EmptyPropertyValueException;
use YooKassa\Common\Exceptions\InvalidPropertyValueException;
use YooKassa\Common\Exceptions\InvalidPropertyValueTypeException;

/**
 * Interface SelfEmployedInterface.
 *
 * @category Interface
 * @package  YooKassa\Model
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 * @property string $id Идентификатор самозанятого в ЮKassa.
 * @property string $status Статус подключения самозанятого и выдачи ЮMoney прав на регистрацию чеков.
 * @property DateTime $created_at Время создания объекта самозанятого.
 * @property DateTime $createdAt Время создания объекта самозанятого.
 * @property null|string $itn ИНН самозанятого.
 * @property null|string $phone Телефон самозанятого, который привязан к личному кабинету в сервисе Мой налог.
 * @property null|SelfEmployedConfirmation $confirmation Сценарий подтверждения пользователем заявки ЮMoney на получение прав для регистрации чеков в сервисе Мой налог.
 * @property bool $test Идентификатор самозанятого в ЮKassa.
 */
interface SelfEmployedInterface
{
    /** @var int Минимальная длина идентификатора */
    public const MIN_LENGTH_ID = 36;

    /** @var int Максимальная длина идентификатора */
    public const MAX_LENGTH_ID = 50;

    /**
     * Возвращает идентификатор самозанятого.
     *
     * @return string|null Идентификатор самозанятого
     */
    public function getId(): ?string;

    /**
     * Устанавливает идентификатор самозанятого.
     *
     * @param string $id Идентификатор самозанятого в ЮKassa
     *
     * @return self
     */
    public function setId(string $id): self;

    /**
     * Возвращает статус самозанятого.
     *
     * @return string|null Статус самозанятого
     */
    public function getStatus(): ?string;

    /**
     * Устанавливает статус самозанятого.
     *
     * @param string $status Статус самозанятого
     *
     * @return $this
     */
    public function setStatus(string $status): self;

    /**
     * Возвращает время создания объекта самозанятого.
     *
     * @return DateTime|null Время создания объекта самозанятого
     */
    public function getCreatedAt(): ?DateTime;

    /**
     * Устанавливает время создания объекта самозанятого.
     *
     * @param DateTime|string|null $created_at Время создания объекта самозанятого
     *
     * @return $this
     */
    public function setCreatedAt(DateTime|string|null $created_at): self;

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
     * @return self
     */
    public function setItn(?string $itn = null): self;

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
     * @return self
     */
    public function setPhone(?string $phone = null): self;

    /**
     * Возвращает сценарий подтверждения.
     */
    public function getConfirmation(): ?SelfEmployedConfirmation;

    /**
     * Устанавливает сценарий подтверждения.
     *
     * @param array|SelfEmployedConfirmation|null $confirmation Сценарий подтверждения
     *
     * @return self
     */
    public function setConfirmation(mixed $confirmation = null): self;

    /**
     * Возвращает признак тестовой операции.
     *
     * @return bool Признак тестовой операции
     */
    public function getTest(): bool;

    /**
     * Устанавливает признак тестовой операции.
     *
     * @param bool $test Признак тестовой операции
     *
     * @return self
     */
    public function setTest(bool $test): self;
}

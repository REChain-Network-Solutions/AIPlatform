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

namespace YooKassa\Request\PersonalData;

use YooKassa\Model\Metadata;

/**
 * Interface CreatePersonalDataRequestInterface.
 *
 * @category Interface
 * @package  YooKassa\Request
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 */
interface CreatePersonalDataRequestInterface
{
    /**
     * Возвращает тип персональных данных.
     *
     * @return string|null Тип персональных данных
     */
    public function getType(): ?string;

    /**
     * Устанавливает тип персональных данных.
     *
     * @param string|null $type Тип персональных данных
     *
     * @return $this
     */
    public function setType(?string $type): self;

    /**
     * Проверяет наличие типа персональных данных в запросе.
     *
     * @return bool True если тип персональных данных задан, false если нет
     */
    public function hasType(): bool;

    /**
     * Возвращает фамилию пользователя.
     *
     * @return string|null Фамилия пользователя
     */
    public function getLastName(): ?string;

    /**
     * Устанавливает фамилию пользователя.
     *
     * @param string|null $last_name Фамилия пользователя
     *
     * @return $this
     */
    public function setLastName(?string $last_name): self;

    /**
     * Проверяет наличие фамилии пользователя в запросе.
     *
     * @return bool True если фамилия пользователя задана, false если нет
     */
    public function hasLastName(): bool;

    /**
     * Возвращает имя пользователя.
     *
     * @return string|null Имя пользователя
     */
    public function getFirstName(): ?string;

    /**
     * Устанавливает имя пользователя.
     *
     * @param string|null $first_name Имя пользователя
     *
     * @return $this
     */
    public function setFirstName(?string $first_name): self;

    /**
     * Проверяет наличие имени пользователя в запросе.
     *
     * @return bool True если имя пользователя задано, false если нет
     */
    public function hasFirstName(): bool;

    /**
     * Возвращает отчество пользователя.
     *
     * @return null|string Отчество пользователя
     */
    public function getMiddleName(): ?string;

    /**
     * Устанавливает отчество пользователя.
     *
     * @param null|string $middle_name Отчество пользователя
     *
     * @return $this
     */
    public function setMiddleName(?string $middle_name = null): self;

    /**
     * Проверяет наличие отчества пользователя в запросе.
     *
     * @return bool True если отчество пользователя задано, false если нет
     */
    public function hasMiddleName(): bool;

    /**
     * Возвращает метаданные.
     *
     * @return Metadata|null Метаданные
     */
    public function getMetadata(): ?Metadata;

    /**
     * Устанавливает метаданные.
     *
     * @param null|array|Metadata $metadata Метаданные
     *
     * @return $this
     */
    public function setMetadata(mixed $metadata): self;

    /**
     * Проверяет, были ли установлены метаданные.
     *
     * @return bool True если метаданные были установлены, false если нет
     */
    public function hasMetadata(): bool;

    /**
     * Проверяет на валидность текущий объект
     *
     * @return bool True если объект запроса валиден, false если нет
     */
    public function validate(): bool;

    /**
     * Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации.
     *
     * @return array Ассоциативный массив со свойствами текущего объекта
     */
    public function toArray(): array;
}

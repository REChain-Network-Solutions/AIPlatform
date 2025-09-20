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

namespace YooKassa\Request\Deals;

use YooKassa\Model\Metadata;

/**
 * Interface CreateDealRequestInterface.
 *
 * @category Interface
 * @package  YooKassa\Request
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 * @property string $type Тип сделки
 * @property string $fee_moment Момент перечисления вознаграждения
 * @property string $feeMoment Момент перечисления вознаграждения
 * @property string $description Описание сделки
 * @property Metadata $metadata Дополнительные данные сделки
 */
interface CreateDealRequestInterface
{
    /**
     * Возвращает тип сделки.
     *
     * @return string|null Тип сделки
     */
    public function getType(): ?string;

    /**
     * Проверяет наличие типа в создаваемой сделке.
     *
     * @return bool True если тип сделки установлен, false если нет
     */
    public function hasType(): bool;

    /**
     * Устанавливает тип сделки.
     *
     * @param string $type Тип сделки
     *
     * @return self
     */
    public function setType(string $type): self;

    /**
     * Возвращает момент перечисления вам вознаграждения платформы.
     *
     * @return string|null Момент перечисления вознаграждения
     */
    public function getFeeMoment(): ?string;

    /**
     * Проверяет наличие момента перечисления вознаграждения в создаваемой сделке.
     *
     * @return bool True если момент перечисления вознаграждения установлен, false если нет
     */
    public function hasFeeMoment(): bool;

    /**
     * Устанавливает момент перечисления вознаграждения платформы.
     *
     * @param string $fee_moment Момент перечисления вознаграждения
     *
     * @return self
     */
    public function setFeeMoment(string $fee_moment): self;

    /**
     * Возвращает описание сделки (не более 128 символов).
     *
     * @return string|null Описание сделки
     */
    public function getDescription(): ?string;

    /**
     * Проверяет наличие описания в создаваемой сделке.
     *
     * @return bool True если описание сделки установлено, false если нет
     */
    public function hasDescription(): bool;

    /**
     * Устанавливает описание сделки.
     *
     * @param string|null $description Описание сделки
     *
     * @return self
     */
    public function setDescription(?string $description): self;

    /**
     * Возвращает дополнительные данные сделки.
     *
     * @return Metadata|null Дополнительные данные сделки
     */
    public function getMetadata(): ?Metadata;

    /**
     * Проверяет, были ли установлены метаданные сделки.
     *
     * @return bool True если метаданные были установлены, false если нет
     */
    public function hasMetadata(): bool;

    /**
     * Устанавливает метаданные, привязанные к сделке.
     *
     * @param null|array|Metadata $metadata Метаданные сделки, устанавливаемые мерчантом
     *
     * @return self
     */
    public function setMetadata(mixed $metadata): self;
}

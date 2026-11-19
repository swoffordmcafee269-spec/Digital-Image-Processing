$ErrorActionPreference = 'Stop'
$revisionDocx = 'D:\数字图像处理实习作业\output\docx\数字图像处理综合实习报告_修订版.docx'
$revisionPdf = 'D:\数字图像处理实习作业\output\word_revision_work\数字图像处理综合实习报告_修订版_排版检查.pdf'
$word = $null
$document = $null
try {
    $word = New-Object -ComObject KWPS.Application
    Write-Output 'Application created'
    $word.Visible = $false
    $word.DisplayAlerts = 0
    Write-Output 'Creating document and inserting revision'
    $document = $word.Documents.Add()
    $word.Selection.InsertFile($revisionDocx)
    Write-Output 'Revision inserted'
    $document.Repaginate()
    Write-Output 'Document repaginated'
    $pages = $document.ComputeStatistics(2)
    Write-Output ('Pages: ' + $pages)
    $document.ExportAsFixedFormat($revisionPdf, 17)
    Write-Output 'PDF exported'
    [pscustomobject]@{Pages=$pages; Pdf=$revisionPdf} | ConvertTo-Json -Compress
} catch {
    Write-Output ('Render error: ' + $_.Exception.Message)
    throw
} finally {
    if ($null -ne $document) { try { $document.Close(0) } catch {} }
    if ($null -ne $word) { try { $word.Quit() } catch {} }
    if ($null -ne $document) { [System.Runtime.InteropServices.Marshal]::ReleaseComObject($document) | Out-Null }
    if ($null -ne $word) { [System.Runtime.InteropServices.Marshal]::ReleaseComObject($word) | Out-Null }
}
